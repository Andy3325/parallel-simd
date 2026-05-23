#include "../PCFG.h"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

static uint64_t UpdateFnv1a(uint64_t hash, const string& value)
{
    const uint64_t prime = 1099511628211ULL;
    for (unsigned char ch : value)
    {
        hash ^= ch;
        hash *= prime;
    }
    hash ^= 0xffU;
    hash *= prime;
    return hash;
}

static string Hex64(uint64_t value)
{
    stringstream ss;
    ss << hex << setw(16) << setfill('0') << value;
    return ss.str();
}

static size_t ActiveSize(const PriorityQueue& q)
{
    if (q.priority_head >= q.priority.size())
    {
        return 0;
    }
    return q.priority.size() - q.priority_head;
}

int main(int argc, char* argv[])
{
    string train_path = "guessdata/benchmark-small.txt";
    int rounds = 1000;

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "--train" && i + 1 < argc)
        {
            train_path = argv[++i];
        }
        else if (arg == "--rounds" && i + 1 < argc)
        {
            rounds = atoi(argv[++i]);
            if (rounds < 0)
            {
                rounds = 0;
            }
        }
    }

    PriorityQueue q;
    q.m.train(train_path);
    q.m.order();
    q.init();

#ifdef ENABLE_PRIORITY_LAZY_OPT
    cout << "MODE|sorted_vector_lazy" << endl;
#else
    cout << "MODE|sorted_vector" << endl;
#endif

    uint64_t checksum = 14695981039346656037ULL;
    size_t total_generated = 0;
    int executed_steps = 0;

    for (int step = 0; step < rounds && ActiveSize(q) > 0; ++step)
    {
        double top_prob = q.priority[q.priority_head].prob;
        size_t active_before = ActiveSize(q);
        size_t guesses_before = q.guesses.size();

        q.PopNext();

        size_t generated = q.guesses.size() - guesses_before;
        for (size_t i = guesses_before; i < q.guesses.size(); ++i)
        {
            checksum = UpdateFnv1a(checksum, q.guesses[i]);
        }
        total_generated += generated;
        executed_steps += 1;

        cout << "TRACE|step=" << step
             << "|top_prob=" << setprecision(9) << top_prob
             << "|generated=" << generated
             << "|total_generated=" << total_generated
             << "|checksum=" << Hex64(checksum)
             << "|active_before=" << active_before
             << "|active_after=" << ActiveSize(q)
             << endl;
    }

    cout << "SUMMARY|popnext_steps=" << executed_steps
         << "|total_generated=" << total_generated
         << "|checksum_hex=" << Hex64(checksum)
         << endl;

    return 0;
}
