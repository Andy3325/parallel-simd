#include "PCFG.h"
#include "guessing_mpi.h"

#include "md5.h"

#include <algorithm>
#include <chrono>

using namespace std;

namespace
{
static void pack_pt(const PT& pt, vector<int>& data, double probs[2])
{
    data.clear();
    data.emplace_back(static_cast<int>(pt.content.size()));
    for (const segment& seg : pt.content)
    {
        data.emplace_back(seg.type);
        data.emplace_back(seg.length);
    }

    data.emplace_back(pt.pivot);

    data.emplace_back(static_cast<int>(pt.curr_indices.size()));
    for (int idx : pt.curr_indices)
    {
        data.emplace_back(idx);
    }

    data.emplace_back(static_cast<int>(pt.max_indices.size()));
    for (int idx : pt.max_indices)
    {
        data.emplace_back(idx);
    }

    probs[0] = pt.preterm_prob;
    probs[1] = pt.prob;
}

static PT unpack_pt(const vector<int>& data, const double probs[2])
{
    PT pt;
    size_t pos = 0;

    const int content_size = data[pos++];
    for (int i = 0; i < content_size; ++i)
    {
        const int type = data[pos++];
        const int length = data[pos++];
        pt.content.emplace_back(type, length);
    }

    pt.pivot = data[pos++];

    const int curr_size = data[pos++];
    for (int i = 0; i < curr_size; ++i)
    {
        pt.curr_indices.emplace_back(data[pos++]);
    }

    const int max_size = data[pos++];
    for (int i = 0; i < max_size; ++i)
    {
        pt.max_indices.emplace_back(data[pos++]);
    }

    pt.preterm_prob = static_cast<float>(probs[0]);
    pt.prob = static_cast<float>(probs[1]);
    return pt;
}

static segment* find_segment(model& m, const segment& seg)
{
    if (seg.type == 1)
    {
        const int id = m.FindLetter(seg);
        return id >= 0 ? &m.letters[id] : nullptr;
    }
    if (seg.type == 2)
    {
        const int id = m.FindDigit(seg);
        return id >= 0 ? &m.digits[id] : nullptr;
    }
    if (seg.type == 3)
    {
        const int id = m.FindSymbol(seg);
        return id >= 0 ? &m.symbols[id] : nullptr;
    }
    return nullptr;
}

static string value_at(model& m, const segment& seg, int value_idx)
{
    segment* found = find_segment(m, seg);
    if (found == nullptr ||
        value_idx < 0 ||
        value_idx >= static_cast<int>(found->ordered_values.size()))
    {
        return "";
    }
    return found->ordered_values[value_idx];
}

static string build_prefix(model& m, const PT& pt)
{
    string prefix;
    const int last_idx = static_cast<int>(pt.content.size()) - 1;
    const int prefix_count =
        min(last_idx, static_cast<int>(pt.curr_indices.size()));

    for (int i = 0; i < prefix_count; ++i)
    {
        prefix += value_at(m, pt.content[i], pt.curr_indices[i]);
    }

    return prefix;
}

static void insert_pt_sorted(PriorityQueue& q, const PT& pt)
{
    if (q.priority.empty())
    {
        q.priority.emplace_back(pt);
        return;
    }

    for (auto iter = q.priority.begin(); iter != q.priority.end(); ++iter)
    {
        if (iter != q.priority.end() - 1 && iter != q.priority.begin())
        {
            if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
            {
                q.priority.emplace(iter + 1, pt);
                return;
            }
        }
        if (iter == q.priority.end() - 1)
        {
            q.priority.emplace_back(pt);
            return;
        }
        if (iter == q.priority.begin() && iter->prob < pt.prob)
        {
            q.priority.emplace(iter, pt);
            return;
        }
    }

    q.priority.emplace_back(pt);
}
}

void BcastPT(PT& pt, int rank)
{
    vector<int> data;
    double probs[2] = {0.0, 0.0};
    int data_size = 0;

    if (rank == 0)
    {
        pack_pt(pt, data, probs);
        data_size = static_cast<int>(data.size());
    }

    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        data.resize(data_size);
    }

    if (data_size > 0)
    {
        MPI_Bcast(data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(probs, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        pt = unpack_pt(data, probs);
    }
}

void BcastPTBatch(vector<PT>& pts, int rank)
{
    vector<int> sizes;
    vector<int> payload;
    vector<double> probs;
    int count = 0;

    if (rank == 0)
    {
        count = static_cast<int>(pts.size());
        sizes.reserve(count);
        probs.reserve(static_cast<size_t>(count) * 2);
        for (const PT& pt : pts)
        {
            vector<int> one;
            double one_probs[2] = {0.0, 0.0};
            pack_pt(pt, one, one_probs);
            sizes.emplace_back(static_cast<int>(one.size()));
            payload.insert(payload.end(), one.begin(), one.end());
            probs.emplace_back(one_probs[0]);
            probs.emplace_back(one_probs[1]);
        }
    }

    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        sizes.resize(count);
        probs.resize(static_cast<size_t>(count) * 2);
    }
    if (count > 0)
    {
        MPI_Bcast(sizes.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(probs.data(), count * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int payload_size = rank == 0 ? static_cast<int>(payload.size()) : 0;
    MPI_Bcast(&payload_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        payload.resize(payload_size);
    }
    if (payload_size > 0)
    {
        MPI_Bcast(payload.data(), payload_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank != 0)
    {
        pts.clear();
        size_t cursor = 0;
        for (int i = 0; i < count; ++i)
        {
            vector<int> one(payload.begin() + cursor,
                            payload.begin() + cursor + sizes[i]);
            const double one_probs[2] = {probs[2 * i], probs[2 * i + 1]};
            pts.emplace_back(unpack_pt(one, one_probs));
            cursor += sizes[i];
        }
    }
}

void AdvanceFrontWithoutGenerate(PriorityQueue& q)
{
    if (q.priority.empty())
    {
        return;
    }

    PT current = q.priority.front();
    vector<PT> new_pts = current.NewPTs();
    for (PT pt : new_pts)
    {
        q.CalProb(pt);
        insert_pt_sorted(q, pt);
    }
    q.priority.erase(q.priority.begin());
}

MPILocalResult GenerateAndHashPTMPI(model& m,
                                    const PT& pt,
                                    const unordered_set<string>& test_set,
                                    int rank,
                                    int world_size)
{
    MPILocalResult result;
    if (pt.content.empty() || pt.max_indices.size() < pt.content.size())
    {
        return result;
    }

    const int last_idx = static_cast<int>(pt.content.size()) - 1;
    segment* last_segment = find_segment(m, pt.content[last_idx]);
    if (last_segment == nullptr)
    {
        return result;
    }

    int n = pt.max_indices[last_idx];
    n = min(n, static_cast<int>(last_segment->ordered_values.size()));
    if (n <= 0)
    {
        return result;
    }

    const int start = n * rank / world_size;
    const int end = n * (rank + 1) / world_size;
    const string prefix = build_prefix(m, pt);

    bit32 state[4] = {0, 0, 0, 0};
    for (int i = start; i < end; ++i)
    {
        const auto gen_start = chrono::steady_clock::now();
        string candidate;
        if (prefix.empty())
        {
            candidate = last_segment->ordered_values[i];
        }
        else
        {
            candidate.reserve(prefix.size() +
                              last_segment->ordered_values[i].size());
            candidate.append(prefix);
            candidate.append(last_segment->ordered_values[i]);
        }
        const auto gen_end = chrono::steady_clock::now();

        const auto hash_start = chrono::steady_clock::now();
        result.generated += 1;
        if (test_set.find(candidate) != test_set.end())
        {
            result.cracked += 1;
        }
        MD5Hash(candidate, state);
        const auto hash_end = chrono::steady_clock::now();

        result.generate_time +=
            chrono::duration<double>(gen_end - gen_start).count();
        result.hash_time +=
            chrono::duration<double>(hash_end - hash_start).count();
    }

    result.compute_time = result.generate_time + result.hash_time;
    return result;
}
