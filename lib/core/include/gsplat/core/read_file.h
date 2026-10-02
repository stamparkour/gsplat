#ifndef GSPLAT_CORE_READ_FILE_H
#define GSPLAT_CORE_READ_FILE_H

#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

namespace gsplat::core {
    std::vector<char> load_entire_file_binary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        std::streamsize size = file.tellg();

        file.seekg(0, std::ios::beg);

       std::vector<char> buffer(size);

       file.read(buffer.data(), size);

        return buffer;
    }
}

#endif // GSPLAT_CORE_READ_FILE_H