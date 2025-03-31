#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "base/unicode.h"
#include "base/unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <locale>
#include <codecvt>

// 计算UTF-8字符的长度(字节数)
size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

// 将Unicode码点数组转换为UTF-8字符串
static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

// 从UTF-8字符串中解析出一个Unicode码点
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    
    // 处理单字节UTF-8字符
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    
    // 检查是否为有效的UTF-8序列开始
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    
    // 处理双字节UTF-8字符
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    
    // 处理三字节UTF-8字符
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    
    // 处理四字节UTF-8字符
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

// Unicode码点标志数组初始化
static std::vector<codepoint_flags> unicode_cpt_flags_array() {
    std::vector<codepoint_flags> cpt_flags(MAX_CODEPOINTS, codepoint_flags::UNDEFINED);

    // 初始化基本标志
    assert(unicode_ranges_flags.front().first == 0);
    assert(unicode_ranges_flags.back().first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags[i-1];  // 范围起始点和标志
        const auto range_end = unicode_ranges_flags[i];    // 范围结束点和标志
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    // 设置特殊字符类型标志
    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

// 构建字节到UTF-8的映射表
static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    
    // 映射ASCII可打印字符
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // '!' 到 '~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    
    // 映射扩展ASCII字符
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // '¡' 到 '¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // '®' 到 'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    
    // 处理剩余字符
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

// 构建UTF-8到字节的映射表
static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    
    // 映射ASCII可打印字符
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // '!' 到 '~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    
    // 映射扩展ASCII字符
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // '¡' 到 '¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // '®' 到 'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    
    // 处理剩余字符
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

// 将UTF-8字符串转换为宽字符串
static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.from_bytes(s);
}

// 对字节编码的单词进行UTF-8处理
static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        // 将单词转换为UTF-8编码
        std::string text_utf;
        auto utf_word = unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        // 对每个字符进行字节编码
        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2系统正则表达式分割实现
// 正则表达式模式: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;  // 存储每个单词的偏移量
    bpe_offsets.reserve(offsets.size());  // 预留大致需要的内存空间

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        // 辅助函数：获取指定位置的码点
        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        // 辅助函数：获取指定位置的码点标志
        auto _get_flags = [&] (const size_t pos) -> codepoint_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos]) : codepoint_flags{};
        };

        // 辅助函数：添加一个token
        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        // 主循环：处理文本中的每个位置
        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // 处理缩写形式：'s|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // 处理字母序列：<space>?\p{L}+
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            // 处理数字序列：<space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            // 处理其他符号序列：<space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            // 处理空白字符
            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // 处理末尾空白：\s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // 处理普通空白：\s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // 处理未匹配的字符
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3系统正则表达式分割实现
// 正则表达式模式: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;  // 存储每个单词的偏移量
    bpe_offsets.reserve(offsets.size());  // 预留大致需要的内存空间

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        // 辅助函数：获取指定位置的码点
        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        // 辅助函数：获取指定位置的码点标志
        auto _get_flags = [&] (const size_t pos) -> codepoint_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos]) : codepoint_flags{};
        };

        // 辅助函数：添加一个token
        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        // 主循环：处理文本中的每个位置
        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // 处理缩写形式（不区分大小写）：(?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // 处理字母序列（可选前缀）：[^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // 处理1-3位数字：\p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // 处理其他符号序列（可选空格前缀和换行后缀）：<space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            // 处理空白字符
            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // 处理换行序列：\s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // 处理末尾空白：\s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // 处理普通空白：\s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // 处理未匹配的字符
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// 使用std::wregex分割文本
static std::vector<size_t> unicode_regex_split_stl(const std::wstring & wtext, const std::wstring & regex_expr, const std::vector<size_t> & offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets;  // 存储每个单词的偏移量
    bpe_offsets.reserve(offsets.size());  // 预留大致需要的内存空间
    
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::wcmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// 使用std::regex分割文本
static std::vector<size_t> unicode_regex_split_stl(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets;  // 存储每个单词的偏移量
    bpe_offsets.reserve(offsets.size());  // 预留大致需要的内存空间
    
    size_t start = 0;
    for (auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::cmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// 使用自定义实现分割文本
static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    // 根据正则表达式模式选择相应的实现
    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    }

    return bpe_offsets;
}

// 公共接口实现

// 将Unicode码点转换为UTF-8字符串
std::string unicode_cpt_to_utf8(uint32_t cp) {
    std::string result;

    if (/* 0x00 <= cp && */ cp <= 0x7f) {
        result.push_back(cp);
        return result;
    }
    if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

// 对Unicode码点数组进行NFD标准化
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.cbegin(), unicode_ranges_nfd.cend(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

// 将UTF-8字符串转换为Unicode码点数组
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}

// 获取Unicode码点的标志
codepoint_flags unicode_cpt_flags(const uint32_t cp) {
    static const codepoint_flags undef(codepoint_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cp < cpt_flags.size() ? cpt_flags[cp] : undef;
}

// 获取UTF-8字符串的标志
codepoint_flags unicode_cpt_flags(const std::string & utf8) {
    static const codepoint_flags undef(codepoint_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // 未定义
    }
    size_t offset = 0;
    return unicode_cpt_flags(unicode_cpt_from_utf8(utf8, offset));
}

// 将字节转换为UTF-8字符串
std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

// 将UTF-8字符串转换为字节
uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

// 将Unicode码点转换为小写
uint32_t unicode_tolower(uint32_t cp) {
    auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}

// 使用正则表达式分割文本
std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    // Unicode类别定义
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", codepoint_flags::NUMBER },      // 数字类别
        { "\\p{L}", codepoint_flags::LETTER },      // 字母类别
        { "\\p{P}", codepoint_flags::PUNCTUATION }, // 标点类别
    };

    // Unicode类别对应的代码点
    static const std::map<int, int> k_ucat_cpt = {
        { codepoint_flags::NUMBER,        0xD1 },
        { codepoint_flags::LETTER,        0xD2 },
        { codepoint_flags::PUNCTUATION,   0xD3 },
    };

    // Unicode类别对应的字符范围
    static const std::map<int, std::string> k_ucat_map = {
        { codepoint_flags::NUMBER,        "\x30-\x39" },                    // 0-9
        { codepoint_flags::LETTER,        "\x41-\x5A\x61-\x7A" },          // A-Za-z
        { codepoint_flags::PUNCTUATION,   "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
    };

    // 检查是否需要进行码点折叠处理
    bool need_collapse = false;
    for (auto & regex_expr : regex_exprs) {
        // 搜索Unicode类别
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    // 生成文本的"折叠"表示，将所有码点替换为单个字节
    std::string text_collapsed;
    if (need_collapse) {
        // 折叠所有Unicode类别
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // 保持单字节码点不变
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags(cpts[i]);

            if (flags.is_whitespace) {
                //注意：C++ std::regex \s 不匹配 0x85，而Rust和Python的regex会匹配
                //text_collapsed[i] = (char) 0x85;  // 使用<Next Line>作为空白字符的替代
                text_collapsed[i] = (char) 0x0B;    // 使用<vertical tab>作为空白字符的替代
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0;  // 默认替代字符
            }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (auto & regex_expr : regex_exprs) {
        // 首先尝试使用高效的自定义正则表达式实现
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // 回退到通用的std::regex / std::wregex实现
        try {
            // 如果正则表达式中使用了Unicode类别，则使用折叠后的文本，并替换Unicode类别为对应的折叠表示
            bool use_collapsed = false;
            for (auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }

            if (use_collapsed) {
                // 检查原始正则表达式是否包含非ASCII字符
                const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("正则表达式同时包含Unicode类别和非ASCII字符 - 不支持此组合");
                    }
                }

                // 生成正则表达式的折叠表示
                std::string regex_expr_collapsed;

                // 跟踪是否在[]内，因为不允许嵌套的[]
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    if (regex_expr[i + 0] == '\\' && i + 4 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{' &&
                        regex_expr[i + 4] == '}') {
                        const std::string pat = regex_expr.substr(i, 5);
                        if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                            if (!inside) {
                                regex_expr_collapsed += '[';
                            }
                            regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                            regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                            if (!inside) {
                                regex_expr_collapsed += ']';
                            }
                            i += 4;
                            continue;
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // 没有使用Unicode类别，可以直接使用std::wregex
                const std::wstring wregex_expr = unicode_wstring_from_utf8(regex_expr);

                // std::wregex的\s不匹配非ASCII空白字符，使用0x0B作为替代
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "处理正则表达式失败: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "正则表达式错误: %s\n", e.what());
            throw std::runtime_error("处理正则表达式失败");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size());  // 预留大致需要的内存空间

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}