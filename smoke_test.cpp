#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include "md5.h"

using namespace std;

static string ToHex(bit32 state[4])
{
    stringstream ss;
    for (int i = 0; i < 4; ++i)
    {
        ss << setw(8) << setfill('0') << hex << state[i];
    }
    return ss.str();
}

int main()
{
    string test_pws[8] = {
        "123456", "password", "12345678", "qwerty",
        "123456789", "12345", "1234", "111111"
    };

    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    // 串行版测试
    for (int i = 0; i < 8; ++i)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        string got = ToHex(state);
        if (got != test_hashes[i])
        {
            cerr << "[FAIL] MD5Hash on " << test_pws[i] << endl;
            cerr << "expected: " << test_hashes[i] << endl;
            cerr << "got:      " << got << endl;
            return 1;
        }
    }

    // 4路接口测试（先测前4个）
    bit32 states4[4][4];
    MD5Hash4(test_pws, states4);
    for (int lane = 0; lane < 4; ++lane)
    {
        string got = ToHex(states4[lane]);
        if (got != test_hashes[lane])
        {
            cerr << "[FAIL] MD5Hash4 lane " << lane << endl;
            cerr << "expected: " << test_hashes[lane] << endl;
            cerr << "got:      " << got << endl;
            return 1;
        }
    }

    cout << "smoke_test passed" << endl;
    return 0;
}
