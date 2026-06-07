#include "PCFG.h"
#include "train_mpi.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

using namespace std;

namespace
{
struct RawValue
{
    long long freq = 0;
    int first_line = numeric_limits<int>::max();
};

struct RawSegment
{
    int type = 0;
    int length = 0;
    long long freq = 0;
    int first_line = numeric_limits<int>::max();
    map<string, RawValue> values;
};

struct RawPT
{
    vector<pair<int, int>> content;
    long long freq = 0;
    int first_line = numeric_limits<int>::max();
};

struct RawModel
{
    long long total_preterm = 0;
    map<string, RawPT> pts;
    map<pair<int, int>, RawSegment> segments;
};

static void append_int(string& buf, int x)
{
    const char* p = reinterpret_cast<const char*>(&x);
    buf.append(p, sizeof(int));
}

static void append_ll(string& buf, long long x)
{
    const char* p = reinterpret_cast<const char*>(&x);
    buf.append(p, sizeof(long long));
}

static void append_string(string& buf, const string& s)
{
    append_int(buf, static_cast<int>(s.size()));
    buf.append(s.data(), s.size());
}

static int read_int(const string& buf, size_t& pos)
{
    int x = 0;
    memcpy(&x, buf.data() + pos, sizeof(int));
    pos += sizeof(int);
    return x;
}

static long long read_ll(const string& buf, size_t& pos)
{
    long long x = 0;
    memcpy(&x, buf.data() + pos, sizeof(long long));
    pos += sizeof(long long);
    return x;
}

static string read_string(const string& buf, size_t& pos)
{
    const int len = read_int(buf, pos);
    string s(buf.data() + pos, len);
    pos += len;
    return s;
}

static void clear_model(model& m)
{
    m.preterm_id = -1;
    m.letters_id = -1;
    m.digits_id = -1;
    m.symbols_id = -1;
    m.total_preterm = 0;
    m.preterminals.clear();
    m.letters.clear();
    m.digits.clear();
    m.symbols.clear();
    m.preterm_freq.clear();
    m.letters_freq.clear();
    m.digits_freq.clear();
    m.symbols_freq.clear();
    m.ordered_pts.clear();
}

static string pt_key(const vector<pair<int, int>>& content)
{
    string key;
    for (const auto& item : content)
    {
        key += to_string(item.first);
        key += ':';
        key += to_string(item.second);
        key += '|';
    }
    return key;
}

static int char_type(unsigned char ch)
{
    if (isalpha(ch))
    {
        return 1;
    }
    if (isdigit(ch))
    {
        return 2;
    }
    return 3;
}

static void add_segment(RawModel& raw,
                        int type,
                        const string& value,
                        int line_id,
                        vector<pair<int, int>>& content)
{
    const int length = static_cast<int>(value.size());
    const pair<int, int> key(type, length);
    RawSegment& seg = raw.segments[key];
    if (seg.freq == 0)
    {
        seg.type = type;
        seg.length = length;
    }
    seg.freq += 1;
    seg.first_line = min(seg.first_line, line_id);

    RawValue& val = seg.values[value];
    val.freq += 1;
    val.first_line = min(val.first_line, line_id);

    content.emplace_back(type, length);
}

static void parse_password_to_raw(RawModel& raw, const string& pw, int line_id)
{
    if (pw.empty())
    {
        return;
    }

    vector<pair<int, int>> content;
    string curr;
    int curr_type = 0;

    for (unsigned char ch : pw)
    {
        const int type = char_type(ch);
        if (curr_type != 0 && type != curr_type)
        {
            add_segment(raw, curr_type, curr, line_id, content);
            curr.clear();
        }
        curr_type = type;
        curr.push_back(static_cast<char>(ch));
    }

    if (!curr.empty())
    {
        add_segment(raw, curr_type, curr, line_id, content);
    }

    raw.total_preterm += 1;
    RawPT& pt = raw.pts[pt_key(content)];
    if (pt.freq == 0)
    {
        pt.content = content;
    }
    pt.freq += 1;
    pt.first_line = min(pt.first_line, line_id);
}

static RawModel train_local_raw(const string& path, int rank, int world_size)
{
    RawModel raw;
    ifstream train_set(path);
    string pw;
    int lines = 0;

    while (train_set >> pw)
    {
        const int line_id = lines;
        lines += 1;

        if (lines % 10000 == 0 && lines > 3000000)
        {
            break;
        }

        if (line_id % world_size == rank)
        {
            parse_password_to_raw(raw, pw, line_id);
        }
    }

    return raw;
}

static string serialize_raw_model(const RawModel& raw)
{
    string buf;
    append_ll(buf, raw.total_preterm);

    append_int(buf, static_cast<int>(raw.pts.size()));
    for (const auto& item : raw.pts)
    {
        const RawPT& pt = item.second;
        append_ll(buf, pt.freq);
        append_int(buf, pt.first_line);
        append_int(buf, static_cast<int>(pt.content.size()));
        for (const auto& seg : pt.content)
        {
            append_int(buf, seg.first);
            append_int(buf, seg.second);
        }
    }

    append_int(buf, static_cast<int>(raw.segments.size()));
    for (const auto& item : raw.segments)
    {
        const RawSegment& seg = item.second;
        append_int(buf, seg.type);
        append_int(buf, seg.length);
        append_ll(buf, seg.freq);
        append_int(buf, seg.first_line);
        append_int(buf, static_cast<int>(seg.values.size()));
        for (const auto& val_item : seg.values)
        {
            append_string(buf, val_item.first);
            append_ll(buf, val_item.second.freq);
            append_int(buf, val_item.second.first_line);
        }
    }

    return buf;
}

static RawModel deserialize_raw_model(const string& buf)
{
    RawModel raw;
    size_t pos = 0;
    raw.total_preterm = read_ll(buf, pos);

    const int pt_count = read_int(buf, pos);
    for (int i = 0; i < pt_count; ++i)
    {
        RawPT pt;
        pt.freq = read_ll(buf, pos);
        pt.first_line = read_int(buf, pos);
        const int content_size = read_int(buf, pos);
        for (int j = 0; j < content_size; ++j)
        {
            const int type = read_int(buf, pos);
            const int length = read_int(buf, pos);
            pt.content.emplace_back(type, length);
        }
        raw.pts[pt_key(pt.content)] = pt;
    }

    const int segment_count = read_int(buf, pos);
    for (int i = 0; i < segment_count; ++i)
    {
        RawSegment seg;
        seg.type = read_int(buf, pos);
        seg.length = read_int(buf, pos);
        seg.freq = read_ll(buf, pos);
        seg.first_line = read_int(buf, pos);
        const int value_count = read_int(buf, pos);
        for (int j = 0; j < value_count; ++j)
        {
            const string value = read_string(buf, pos);
            RawValue rv;
            rv.freq = read_ll(buf, pos);
            rv.first_line = read_int(buf, pos);
            seg.values[value] = rv;
        }
        raw.segments[make_pair(seg.type, seg.length)] = seg;
    }

    return raw;
}

static void merge_raw_model(RawModel& dst, const RawModel& src)
{
    dst.total_preterm += src.total_preterm;

    for (const auto& item : src.pts)
    {
        const RawPT& src_pt = item.second;
        RawPT& dst_pt = dst.pts[item.first];
        if (dst_pt.freq == 0)
        {
            dst_pt.content = src_pt.content;
        }
        dst_pt.freq += src_pt.freq;
        dst_pt.first_line = min(dst_pt.first_line, src_pt.first_line);
    }

    for (const auto& item : src.segments)
    {
        const RawSegment& src_seg = item.second;
        RawSegment& dst_seg = dst.segments[item.first];
        if (dst_seg.freq == 0)
        {
            dst_seg.type = src_seg.type;
            dst_seg.length = src_seg.length;
        }
        dst_seg.freq += src_seg.freq;
        dst_seg.first_line = min(dst_seg.first_line, src_seg.first_line);

        for (const auto& value_item : src_seg.values)
        {
            RawValue& dst_value = dst_seg.values[value_item.first];
            dst_value.freq += value_item.second.freq;
            dst_value.first_line =
                min(dst_value.first_line, value_item.second.first_line);
        }
    }
}

static void add_segment_to_model(model& m, const RawSegment& raw_seg)
{
    segment seg(raw_seg.type, raw_seg.length);

    vector<pair<string, RawValue>> values(raw_seg.values.begin(),
                                          raw_seg.values.end());
    sort(values.begin(), values.end(),
         [](const pair<string, RawValue>& a, const pair<string, RawValue>& b)
         {
             if (a.second.first_line != b.second.first_line)
             {
                 return a.second.first_line < b.second.first_line;
             }
             return a.first < b.first;
         });

    for (const auto& value_item : values)
    {
        const int id = static_cast<int>(seg.values.size());
        seg.values[value_item.first] = id;
        seg.freqs[id] = static_cast<int>(value_item.second.freq);
    }

    int id = 0;
    if (raw_seg.type == 1)
    {
        id = static_cast<int>(m.letters.size());
        m.letters.emplace_back(seg);
        m.letters_freq[id] = static_cast<int>(raw_seg.freq);
    }
    else if (raw_seg.type == 2)
    {
        id = static_cast<int>(m.digits.size());
        m.digits.emplace_back(seg);
        m.digits_freq[id] = static_cast<int>(raw_seg.freq);
    }
    else
    {
        id = static_cast<int>(m.symbols.size());
        m.symbols.emplace_back(seg);
        m.symbols_freq[id] = static_cast<int>(raw_seg.freq);
    }
    (void)id;
}

static model build_model_from_raw(const RawModel& raw)
{
    model m;
    clear_model(m);
    m.total_preterm = static_cast<int>(raw.total_preterm);

    vector<RawPT> pts;
    for (const auto& item : raw.pts)
    {
        pts.emplace_back(item.second);
    }
    sort(pts.begin(), pts.end(),
         [](const RawPT& a, const RawPT& b)
         {
             if (a.first_line != b.first_line)
             {
                 return a.first_line < b.first_line;
             }
             return pt_key(a.content) < pt_key(b.content);
         });

    for (const RawPT& raw_pt : pts)
    {
        PT pt;
        for (const auto& seg : raw_pt.content)
        {
            pt.content.emplace_back(seg.first, seg.second);
            pt.curr_indices.emplace_back(0);
        }
        const int id = static_cast<int>(m.preterminals.size());
        m.preterminals.emplace_back(pt);
        m.preterm_freq[id] = static_cast<int>(raw_pt.freq);
    }

    vector<RawSegment> segments;
    for (const auto& item : raw.segments)
    {
        segments.emplace_back(item.second);
    }
    sort(segments.begin(), segments.end(),
         [](const RawSegment& a, const RawSegment& b)
         {
             if (a.type != b.type)
             {
                 return a.type < b.type;
             }
             if (a.first_line != b.first_line)
             {
                 return a.first_line < b.first_line;
             }
             return a.length < b.length;
         });

    for (const RawSegment& seg : segments)
    {
        add_segment_to_model(m, seg);
    }

    m.preterm_id = static_cast<int>(m.preterminals.size()) - 1;
    m.letters_id = static_cast<int>(m.letters.size()) - 1;
    m.digits_id = static_cast<int>(m.digits.size()) - 1;
    m.symbols_id = static_cast<int>(m.symbols.size()) - 1;
    return m;
}

static void serialize_pt_model(string& buf, const PT& pt, int freq)
{
    append_int(buf, static_cast<int>(pt.content.size()));
    for (const segment& seg : pt.content)
    {
        append_int(buf, seg.type);
        append_int(buf, seg.length);
    }
    append_int(buf, freq);
}

static PT deserialize_pt_model(const string& buf, size_t& pos, int& freq)
{
    PT pt;
    const int content_size = read_int(buf, pos);
    for (int i = 0; i < content_size; ++i)
    {
        const int type = read_int(buf, pos);
        const int length = read_int(buf, pos);
        pt.content.emplace_back(type, length);
        pt.curr_indices.emplace_back(0);
    }
    freq = read_int(buf, pos);
    return pt;
}

static void serialize_segment_model(string& buf,
                                    const segment& seg,
                                    int seg_freq)
{
    append_int(buf, seg.type);
    append_int(buf, seg.length);
    append_int(buf, seg_freq);
    append_int(buf, static_cast<int>(seg.values.size()));

    vector<pair<int, string>> by_id;
    by_id.reserve(seg.values.size());
    for (const auto& item : seg.values)
    {
        by_id.emplace_back(item.second, item.first);
    }
    sort(by_id.begin(), by_id.end());

    for (const auto& item : by_id)
    {
        append_string(buf, item.second);
        append_int(buf, seg.freqs.at(item.first));
    }
}

static segment deserialize_segment_model(const string& buf,
                                         size_t& pos,
                                         int& seg_freq)
{
    const int type = read_int(buf, pos);
    const int length = read_int(buf, pos);
    seg_freq = read_int(buf, pos);
    segment seg(type, length);

    const int value_count = read_int(buf, pos);
    for (int i = 0; i < value_count; ++i)
    {
        const string value = read_string(buf, pos);
        const int freq = read_int(buf, pos);
        seg.values[value] = i;
        seg.freqs[i] = freq;
    }

    return seg;
}

static string serialize_model_raw_order(const model& m)
{
    string buf;
    append_int(buf, m.total_preterm);

    append_int(buf, static_cast<int>(m.preterminals.size()));
    for (int i = 0; i < static_cast<int>(m.preterminals.size()); ++i)
    {
        serialize_pt_model(buf, m.preterminals[i], m.preterm_freq.at(i));
    }

    append_int(buf, static_cast<int>(m.letters.size()));
    for (int i = 0; i < static_cast<int>(m.letters.size()); ++i)
    {
        serialize_segment_model(buf, m.letters[i], m.letters_freq.at(i));
    }

    append_int(buf, static_cast<int>(m.digits.size()));
    for (int i = 0; i < static_cast<int>(m.digits.size()); ++i)
    {
        serialize_segment_model(buf, m.digits[i], m.digits_freq.at(i));
    }

    append_int(buf, static_cast<int>(m.symbols.size()));
    for (int i = 0; i < static_cast<int>(m.symbols.size()); ++i)
    {
        serialize_segment_model(buf, m.symbols[i], m.symbols_freq.at(i));
    }

    return buf;
}

static model deserialize_model_raw_order(const string& buf)
{
    model m;
    clear_model(m);
    size_t pos = 0;

    m.total_preterm = read_int(buf, pos);

    const int pt_count = read_int(buf, pos);
    for (int i = 0; i < pt_count; ++i)
    {
        int freq = 0;
        PT pt = deserialize_pt_model(buf, pos, freq);
        m.preterminals.emplace_back(pt);
        m.preterm_freq[i] = freq;
    }

    const int letter_count = read_int(buf, pos);
    for (int i = 0; i < letter_count; ++i)
    {
        int freq = 0;
        segment seg = deserialize_segment_model(buf, pos, freq);
        m.letters.emplace_back(seg);
        m.letters_freq[i] = freq;
    }

    const int digit_count = read_int(buf, pos);
    for (int i = 0; i < digit_count; ++i)
    {
        int freq = 0;
        segment seg = deserialize_segment_model(buf, pos, freq);
        m.digits.emplace_back(seg);
        m.digits_freq[i] = freq;
    }

    const int symbol_count = read_int(buf, pos);
    for (int i = 0; i < symbol_count; ++i)
    {
        int freq = 0;
        segment seg = deserialize_segment_model(buf, pos, freq);
        m.symbols.emplace_back(seg);
        m.symbols_freq[i] = freq;
    }

    m.preterm_id = static_cast<int>(m.preterminals.size()) - 1;
    m.letters_id = static_cast<int>(m.letters.size()) - 1;
    m.digits_id = static_cast<int>(m.digits.size()) - 1;
    m.symbols_id = static_cast<int>(m.symbols.size()) - 1;
    return m;
}
}

void train_mpi(model& m, const string& path, int rank, int world_size)
{
    clear_model(m);

    const RawModel local_raw = train_local_raw(path, rank, world_size);
    const string local_buf = serialize_raw_model(local_raw);
    const int local_size = static_cast<int>(local_buf.size());

    vector<int> recv_sizes;
    vector<int> displs;
    if (rank == 0)
    {
        recv_sizes.resize(world_size);
    }

    MPI_Gather(&local_size, 1, MPI_INT,
               rank == 0 ? recv_sizes.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    string all_buf;
    if (rank == 0)
    {
        displs.resize(world_size);
        int total_size = 0;
        for (int i = 0; i < world_size; ++i)
        {
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }
        all_buf.resize(total_size);
    }

    MPI_Gatherv(local_buf.data(), local_size, MPI_CHAR,
                rank == 0 && !all_buf.empty() ? &all_buf[0] : nullptr,
                rank == 0 ? recv_sizes.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_CHAR, 0, MPI_COMM_WORLD);

    string global_buf;
    if (rank == 0)
    {
        RawModel global_raw;
        for (int i = 0; i < world_size; ++i)
        {
            const string one_buf(all_buf.data() + displs[i], recv_sizes[i]);
            merge_raw_model(global_raw, deserialize_raw_model(one_buf));
        }
        m = build_model_from_raw(global_raw);
        global_buf = serialize_model_raw_order(m);
    }

    int global_size = rank == 0 ? static_cast<int>(global_buf.size()) : 0;
    MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        global_buf.resize(global_size);
    }
    if (global_size > 0)
    {
        MPI_Bcast(&global_buf[0], global_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    if (rank != 0)
    {
        m = deserialize_model_raw_order(global_buf);
    }

    m.order();
}
