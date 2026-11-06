#include <gtest/gtest.h>
#include <gsplat/data/epoch_file.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

using namespace gsplat::data;

namespace {
	// Builds a set of 'count' gaussians where every value depends on the index,
	// so a mix-up in ordering or a dropped field shows up as a failed comparison.
	gaussian_set make_set(int count) {
		gaussian_set set;
		for (int i = 0; i < count; i++) {
			float f = static_cast<float>(i);
			gaussian g{};
			g.covariance_inv = glm::mat4(f);
			g.covariance_inv[1][2] = 0.5f * f;
			g.mean = {f, f + 1.0f, f + 2.0f, 0.0f};
			g.quaternion = {0.0f, 0.0f, 0.0f, 1.0f};
			g.scale = {0.1f, 0.2f, 0.3f, 0.0f};
			g.color.color = {0.1f, 0.2f, 0.3f, f * 1e-4f};
			set.data().push_back(g);
		}
		return set;
	}

	bool same(const gaussian& a, const gaussian& b) {
		return a.covariance_inv == b.covariance_inv
			&& a.mean == b.mean
			&& a.quaternion == b.quaternion
			&& a.scale == b.scale
			&& a.color.color == b.color.color;
	}

	class EpochFileTest : public ::testing::Test {
	protected:
		std::filesystem::path dir;

		void SetUp() override {
			dir = std::filesystem::temp_directory_path() / "gsplat_epoch_file_test";
			std::filesystem::create_directories(dir);
		}
		void TearDown() override {
			std::filesystem::remove_all(dir);
		}
		std::filesystem::path file(const std::string& name) const { return dir / name; }
	};
}

TEST_F(EpochFileTest, RoundTripKeepsEveryGaussian) {
	gaussian_set original = make_set(1000);
	save_bin(original, file("a.gsplat"));

	gaussian_set loaded;
	load_bin(loaded, file("a.gsplat"));

	ASSERT_EQ(loaded.size(), original.size());
	for (std::size_t i = 0; i < original.size(); i++) {
		EXPECT_TRUE(same(original.data()[i], loaded.data()[i])) << "gaussian " << i << " differs";
	}
}

TEST_F(EpochFileTest, MetadataRoundTrip) {
	save_bin(make_set(10), file("m.gsplat"), epoch_metadata{42, 0.125f, "garden scene"});

	gaussian_set loaded;
	epoch_metadata meta;
	load_bin(loaded, file("m.gsplat"), &meta);

	EXPECT_EQ(meta.iteration, 42u);
	EXPECT_EQ(meta.loss, 0.125f);
	EXPECT_EQ(meta.description, "garden scene");
}

TEST_F(EpochFileTest, EmptySetRoundTrip) {
	save_bin(gaussian_set{}, file("empty.gsplat"));

	gaussian_set loaded = make_set(5); // must be replaced, not appended to
	load_bin(loaded, file("empty.gsplat"));
	EXPECT_EQ(loaded.size(), 0u);
}

TEST_F(EpochFileTest, LoadReplacesExistingContents) {
	save_bin(make_set(3), file("three.gsplat"));

	gaussian_set loaded = make_set(100);
	load_bin(loaded, file("three.gsplat"));
	EXPECT_EQ(loaded.size(), 3u);
}

TEST_F(EpochFileTest, SaveOverwritesExistingFile) {
	save_bin(make_set(10), file("o.gsplat"));
	save_bin(make_set(4), file("o.gsplat"));

	gaussian_set loaded;
	load_bin(loaded, file("o.gsplat"));
	EXPECT_EQ(loaded.size(), 4u);
	EXPECT_FALSE(std::filesystem::exists(file("o.gsplat.tmp"))); // temp file cleaned up
}

TEST_F(EpochFileTest, MissingFileThrows) {
	gaussian_set loaded = make_set(2);
	EXPECT_THROW(load_bin(loaded, file("does_not_exist.gsplat")), std::runtime_error);
	EXPECT_EQ(loaded.size(), 2u); // untouched
}

TEST_F(EpochFileTest, WrongFormatThrows) {
	{
		std::ofstream f(file("bad.gsplat"), std::ios::binary);
		f << "this is definitely not a gsplat file";
	}
	gaussian_set loaded = make_set(2);
	EXPECT_THROW(load_bin(loaded, file("bad.gsplat")), std::runtime_error);
	EXPECT_EQ(loaded.size(), 2u);
}

TEST_F(EpochFileTest, TruncatedFileThrows) {
	save_bin(make_set(100), file("full.gsplat"));

	std::string bytes;
	{
		std::ifstream in(file("full.gsplat"), std::ios::binary);
		bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	}
	{
		std::ofstream out(file("cut.gsplat"), std::ios::binary);
		out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
	}

	gaussian_set loaded = make_set(2);
	EXPECT_THROW(load_bin(loaded, file("cut.gsplat")), std::runtime_error);
	EXPECT_EQ(loaded.size(), 2u);
}
