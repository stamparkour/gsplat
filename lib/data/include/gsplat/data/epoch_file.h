#ifndef GSPLAT_DATA_EPOCH_FILE_H
#define GSPLAT_DATA_EPOCH_FILE_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian_set.h>
#include <cstdint>
#include <filesystem>
#include <string>

namespace gsplat::data {
	// Version of the on-disk format written by save_bin.
	// Bump this whenever the saved layout changes (e.g. a new member in gaussian).
	inline constexpr std::uint32_t epoch_file_version = 1;

	// Extra information stored next to the gaussians.
	struct epoch_metadata {
		std::uint64_t iteration = 0;  // training iteration / epoch this snapshot was taken at
		float loss = 0.0f;            // training loss at that point (0 if unknown)
		std::string description;      // free text, e.g. dataset name or notes

		template <class Archive>
		void serialize(Archive& ar) {
			ar(iteration, loss, description);
		}
	};

	// Writes the CPU-side gaussians (set.data()) to a binary file.
	// NOTE: this does not read back from the GPU. Call set.fetch_vulkan() first
	// if the newest values live in the Vulkan buffer.
	// Throws std::runtime_error on failure.
	void save_bin(const gaussian_set& set, const std::filesystem::path& path, const epoch_metadata& meta = {});

	// Replaces the contents of set.data() with the gaussians from the file.
	// If 'meta' is not null it receives the stored metadata.
	// NOTE: this does not upload to the GPU. Call set.store_vulkan() afterwards
	// (and make sure the Vulkan buffers have the right size).
	// If anything fails, an exception is thrown and 'set' is left unchanged.
	void load_bin(gaussian_set& set, const std::filesystem::path& path, epoch_metadata* meta = nullptr);
}

#endif // GSPLAT_DATA_EPOCH_FILE_H
