#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif //为了本地编译通过desu

using namespace std;
using namespace chrono;

static inline uint32_t ByteSwap32(uint32_t value)
{
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8) |
           ((value & 0x00ff0000u) >> 8) |
           ((value & 0xff000000u) >> 24);
}

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
    // 将输入的字符串转换为Byte为单位的数组
    Byte *blocks = (Byte *)input.c_str();
    int length = input.length();

    // 计算原始消息长度（以比特为单位）
    int bitLength = length * 8;

    // paddingBits: 原始消息需要的padding长度（以bit为单位）
    // 对于给定的消息，将其补齐至length%512==448为止
    // 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
    {
        paddingBits += 512 - (paddingBits - 448);
    }
    else if (paddingBits < 448)
    {
        paddingBits = 448 - paddingBits;
    }
    else if (paddingBits == 448)
    {
        paddingBits = 512;
    }

    // 原始消息需要的padding长度（以Byte为单位）
    int paddingBytes = paddingBits / 8;
    // 创建最终的字节数组
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节。填充时，第一位为1，后面的所有位均为0。
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);

    // 添加消息长度（64比特，小端格式）
    for (int i = 0; i < 8; ++i)
    {
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    // 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
    *n_byte = paddedLength;
    return paddedMessage;
}

/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{
    Byte *paddedMessage;
    int *messageLength = new int[1];
    paddedMessage = StringProcess(input, &messageLength[0]);
    int n_blocks = messageLength[0] / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // 逐block地更新state
    for (int i = 0; i < n_blocks; i += 1)
    {
        bit32 x[16];

        for (int i1 = 0; i1 < 16; ++i1)
        {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                    (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                    (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                    (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        /* Round 1 */
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x02441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x04881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    for (int i = 0; i < 4; i++)
    {
        state[i] = ByteSwap32(state[i]);
    }

    delete[] paddedMessage;
    delete[] messageLength;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#define VF(x, y, z) vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32((x)), (z)))
#define VG(x, y, z) vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32((z))))
#define VH(x, y, z) veorq_u32(veorq_u32((x), (y)), (z))
#define VI(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32((z))))
#define VROTATELEFT(num, n) vorrq_u32(vshlq_n_u32((num), (n)), vshrq_n_u32((num), 32 - (n)))

#define VFF(a, b, c, d, x, s, ac) do { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(VF((b), (c), (d)), (x)), vdupq_n_u32((ac)))); \
    (a) = VROTATELEFT((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while (0)

#define VGG(a, b, c, d, x, s, ac) do { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(VG((b), (c), (d)), (x)), vdupq_n_u32((ac)))); \
    (a) = VROTATELEFT((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while (0)

#define VHH(a, b, c, d, x, s, ac) do { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(VH((b), (c), (d)), (x)), vdupq_n_u32((ac)))); \
    (a) = VROTATELEFT((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while (0)

#define VII(a, b, c, d, x, s, ac) do { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(VI((b), (c), (d)), (x)), vdupq_n_u32((ac)))); \
    (a) = VROTATELEFT((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while (0)
#endif

void MD5Hash4(const string *inputs, bit32 states[4][4])
{
#if defined(__ARM_NEON) || defined(__aarch64__)
    Byte *paddedMessages[4] = {nullptr, nullptr, nullptr, nullptr};
    int messageLengths[4] = {0, 0, 0, 0};
    int blockCounts[4] = {0, 0, 0, 0};
    int maxBlocks = 0;

    for (int lane = 0; lane < 4; ++lane)
    {
        paddedMessages[lane] = StringProcess(inputs[lane], &messageLengths[lane]);
        blockCounts[lane] = messageLengths[lane] / 64;
        if (blockCounts[lane] > maxBlocks)
        {
            maxBlocks = blockCounts[lane];
        }
    }

    uint32x4_t state0 = vdupq_n_u32(0x67452301u);
    uint32x4_t state1 = vdupq_n_u32(0xefcdab89u);
    uint32x4_t state2 = vdupq_n_u32(0x98badcfeu);
    uint32x4_t state3 = vdupq_n_u32(0x10325476u);

    for (int block = 0; block < maxBlocks; ++block)
    {
        uint32_t activeMaskArr[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            activeMaskArr[lane] = (block < blockCounts[lane]) ? 0xffffffffu : 0u;
        }
        uint32x4_t activeMask = vld1q_u32(activeMaskArr);

        uint32x4_t x[16];
        for (int word = 0; word < 16; ++word)
        {
            uint32_t lanes[4] = {0u, 0u, 0u, 0u};
            for (int lane = 0; lane < 4; ++lane)
            {
                if (block < blockCounts[lane])
                {
                    Byte *base = paddedMessages[lane] + block * 64 + word * 4;
                    lanes[lane] = (uint32_t(base[0])) |
                                  (uint32_t(base[1]) << 8) |
                                  (uint32_t(base[2]) << 16) |
                                  (uint32_t(base[3]) << 24);
                }
            }
            x[word] = vld1q_u32(lanes);
        }

        uint32x4_t a = state0;
        uint32x4_t b = state1;
        uint32x4_t c = state2;
        uint32x4_t d = state3;

        /* Round 1 */
        VFF(a, b, c, d, x[0], s11, 0xd76aa478u);
        VFF(d, a, b, c, x[1], s12, 0xe8c7b756u);
        VFF(c, d, a, b, x[2], s13, 0x242070dbu);
        VFF(b, c, d, a, x[3], s14, 0xc1bdceeeu);
        VFF(a, b, c, d, x[4], s11, 0xf57c0fafu);
        VFF(d, a, b, c, x[5], s12, 0x4787c62au);
        VFF(c, d, a, b, x[6], s13, 0xa8304613u);
        VFF(b, c, d, a, x[7], s14, 0xfd469501u);
        VFF(a, b, c, d, x[8], s11, 0x698098d8u);
        VFF(d, a, b, c, x[9], s12, 0x8b44f7afu);
        VFF(c, d, a, b, x[10], s13, 0xffff5bb1u);
        VFF(b, c, d, a, x[11], s14, 0x895cd7beu);
        VFF(a, b, c, d, x[12], s11, 0x6b901122u);
        VFF(d, a, b, c, x[13], s12, 0xfd987193u);
        VFF(c, d, a, b, x[14], s13, 0xa679438eu);
        VFF(b, c, d, a, x[15], s14, 0x49b40821u);

        /* Round 2 */
        VGG(a, b, c, d, x[1], s21, 0xf61e2562u);
        VGG(d, a, b, c, x[6], s22, 0xc040b340u);
        VGG(c, d, a, b, x[11], s23, 0x265e5a51u);
        VGG(b, c, d, a, x[0], s24, 0xe9b6c7aau);
        VGG(a, b, c, d, x[5], s21, 0xd62f105du);
        VGG(d, a, b, c, x[10], s22, 0x02441453u);
        VGG(c, d, a, b, x[15], s23, 0xd8a1e681u);
        VGG(b, c, d, a, x[4], s24, 0xe7d3fbc8u);
        VGG(a, b, c, d, x[9], s21, 0x21e1cde6u);
        VGG(d, a, b, c, x[14], s22, 0xc33707d6u);
        VGG(c, d, a, b, x[3], s23, 0xf4d50d87u);
        VGG(b, c, d, a, x[8], s24, 0x455a14edu);
        VGG(a, b, c, d, x[13], s21, 0xa9e3e905u);
        VGG(d, a, b, c, x[2], s22, 0xfcefa3f8u);
        VGG(c, d, a, b, x[7], s23, 0x676f02d9u);
        VGG(b, c, d, a, x[12], s24, 0x8d2a4c8au);

        /* Round 3 */
        VHH(a, b, c, d, x[5], s31, 0xfffa3942u);
        VHH(d, a, b, c, x[8], s32, 0x8771f681u);
        VHH(c, d, a, b, x[11], s33, 0x6d9d6122u);
        VHH(b, c, d, a, x[14], s34, 0xfde5380cu);
        VHH(a, b, c, d, x[1], s31, 0xa4beea44u);
        VHH(d, a, b, c, x[4], s32, 0x4bdecfa9u);
        VHH(c, d, a, b, x[7], s33, 0xf6bb4b60u);
        VHH(b, c, d, a, x[10], s34, 0xbebfbc70u);
        VHH(a, b, c, d, x[13], s31, 0x289b7ec6u);
        VHH(d, a, b, c, x[0], s32, 0xeaa127fau);
        VHH(c, d, a, b, x[3], s33, 0xd4ef3085u);
        VHH(b, c, d, a, x[6], s34, 0x04881d05u);
        VHH(a, b, c, d, x[9], s31, 0xd9d4d039u);
        VHH(d, a, b, c, x[12], s32, 0xe6db99e5u);
        VHH(c, d, a, b, x[15], s33, 0x1fa27cf8u);
        VHH(b, c, d, a, x[2], s34, 0xc4ac5665u);

        /* Round 4 */
        VII(a, b, c, d, x[0], s41, 0xf4292244u);
        VII(d, a, b, c, x[7], s42, 0x432aff97u);
        VII(c, d, a, b, x[14], s43, 0xab9423a7u);
        VII(b, c, d, a, x[5], s44, 0xfc93a039u);
        VII(a, b, c, d, x[12], s41, 0x655b59c3u);
        VII(d, a, b, c, x[3], s42, 0x8f0ccc92u);
        VII(c, d, a, b, x[10], s43, 0xffeff47du);
        VII(b, c, d, a, x[1], s44, 0x85845dd1u);
        VII(a, b, c, d, x[8], s41, 0x6fa87e4fu);
        VII(d, a, b, c, x[15], s42, 0xfe2ce6e0u);
        VII(c, d, a, b, x[6], s43, 0xa3014314u);
        VII(b, c, d, a, x[13], s44, 0x4e0811a1u);
        VII(a, b, c, d, x[4], s41, 0xf7537e82u);
        VII(d, a, b, c, x[11], s42, 0xbd3af235u);
        VII(c, d, a, b, x[2], s43, 0x2ad7d2bbu);
        VII(b, c, d, a, x[9], s44, 0xeb86d391u);

        uint32x4_t candidate0 = vaddq_u32(state0, a);
        uint32x4_t candidate1 = vaddq_u32(state1, b);
        uint32x4_t candidate2 = vaddq_u32(state2, c);
        uint32x4_t candidate3 = vaddq_u32(state3, d);

        state0 = vbslq_u32(activeMask, candidate0, state0);
        state1 = vbslq_u32(activeMask, candidate1, state1);
        state2 = vbslq_u32(activeMask, candidate2, state2);
        state3 = vbslq_u32(activeMask, candidate3, state3);
    }

    uint32_t out0[4], out1[4], out2[4], out3[4];
    vst1q_u32(out0, state0);
    vst1q_u32(out1, state1);
    vst1q_u32(out2, state2);
    vst1q_u32(out3, state3);

    for (int lane = 0; lane < 4; ++lane)
    {
        states[lane][0] = ByteSwap32(out0[lane]);
        states[lane][1] = ByteSwap32(out1[lane]);
        states[lane][2] = ByteSwap32(out2[lane]);
        states[lane][3] = ByteSwap32(out3[lane]);
        delete[] paddedMessages[lane];
    }
#else
    for (int lane = 0; lane < 4; ++lane)
    {
        MD5Hash(inputs[lane], states[lane]);
    }
#endif
}