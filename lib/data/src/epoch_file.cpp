#include <gsplat/data/epoch_file.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace gsplat::data;

/*
File layout (all written through cereal's binary archive):

	char[4]   magic     "GSPL"
	uint32    version   epoch_file_version
	epoch_metadata      iteration, loss, description
	vector<gaussian>    element count, then every gaussian

The magic + version let load_bin reject files that are not ours, or that were
written by a newer/older incompatible version, instead of reading garbage.
*/

namespace {
	constexpr char file_magic[4] = {'G', 'S', 'P', 'L'};
}

void gsplat::data::save_bin(const gaussian_set& set, const std::filesystem::path& path, const epoch_metadata& meta) {
	// Write to a temporary file first and rename at the end, so a crash halfway
	// through never leaves a half-written file in place of a good checkpoint.
	std::filesystem::path tmp = path;
	tmp += ".tmp";

	{
		std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
		if (!out) throw std::runtime_error("save_bin: cannot open " + tmp.string());

		cereal::BinaryOutputArchive ar{out};
		ar(cereal::binary_data(file_magic, sizeof(file_magic)));
		ar(epoch_file_version);
		ar(meta);
		ar(set.data());

		out.flush();
		if (!out) throw std::runtime_error("save_bin: write failed for " + tmp.string());
	} // archive and stream are closed here, before the rename

	std::filesystem::rename(tmp, path); // replaces an existing file
}

void gsplat::data::load_bin(gaussian_set& set, const std::filesystem::path& path, epoch_metadata* meta) {
	std::ifstream in{path, std::ios::binary};
	if (!in) throw std::runtime_error("load_bin: cannot open " + path.string());

	// Read into temporaries so 'set' is untouched if anything goes wrong.
	epoch_metadata loaded_meta{};
	std::vector<gaussian> loaded;

	try {
		cereal::BinaryInputArchive ar{in};

		char magic[4] = {};
		ar(cereal::binary_data(magic, sizeof(magic)));
		if (std::string{magic, 4} != std::string{file_magic, 4})
			throw std::runtime_error("load_bin: not a gsplat epoch file: " + path.string());

		std::uint32_t version = 0;
		ar(version);
		if (version != epoch_file_version)
			throw std::runtime_error("load_bin: unsupported file version " + std::to_string(version) +
				" (this build reads version " + std::to_string(epoch_file_version) + ")");

		ar(loaded_meta);
		ar(loaded);
	}
	catch (const cereal::Exception& e) {
		// cereal throws its own exception type on truncated / corrupt input
		throw std::runtime_error(std::string{"load_bin: corrupt or truncated file: "} + e.what());
	}

	set.data() = std::move(loaded);
	if (meta) *meta = std::move(loaded_meta);
}
