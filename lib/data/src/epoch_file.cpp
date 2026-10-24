#include <gsplat/data/epoch_file.h>
#include <filesystem>
#include <cereal/archives/binary.hpp>

using namespace gsplat::data;

/*

requirements:
- save gaussian set
- load gaassian set
- add metadata to file

optional:
- file version info

you may edit:
- lib/data/src/epoch_file.cpp (all implementation and behavior here)
- lib/data/include/gplat/data/epoch_file.h (all declarations and api here)
- lib/data/include/gplat/data/gaussian_set.h (edit to add c++ reflections for cereal)
- lib/data/include/gplat/data/gaussian.h (edit to add c++ reflections for cereal)
- lib/data/include/gplat/data/gaussian_color.h (edit to add c++ reflections for cereal)

*/


// https://www.learncpp.com/ learn cpp
// https://cppreference.com/ c++ documentation


void save_bin(const gaussian_set& set, const std::filesystem::path& path) {
	// use cereal to save the binary.
	// 'set.data()' is the source storage.
	// https://uscilab.github.io/cereal/quickstart.html
}
void load_bin(gaussian_set& set, const std::filesystem::path& path) {
	// use cereal to save the binary.
	// 'set.data()' is the source storage.
	// https://uscilab.github.io/cereal/quickstart.html
}