#ifndef GSPLAT_TRAIN_MODIFY_H
#define GSPLAT_TRAIN_MODIFY_H

#include <gsplat/data/database.h>
#include <random>
#include <span>
#include <vector>

// adaptive density control, section 5.2 of the 3DGS paper (Kerbl et al. 2023)
namespace gsplat::train {
	constexpr float tau_pos = 0.0002f;      // view space gradient above this gets cloned or split
	constexpr float epsilon_alpha = 0.005f; // alpha below this gets removed
	constexpr float percent_dense = 0.01f;  // clone at or below this times the scene extent, split above
	constexpr float phi = 1.6f;             // split children get scale / phi
	// epsilon_alpha and percent_dense come from the paper's code, the text doesn't give numbers

	// grad[i] is gaussian i's view space position gradient, averaged over the views that saw it
	void split_gaussians(std::vector<gsplat::data::gaussian>& gaussians, std::span<const float> grad, float scene_extent, std::mt19937& rng);
	void remove_gaussians(std::vector<gsplat::data::gaussian>& gaussians);

	// pulls the gaussians off the GPU, splits, removes, and puts them back at the new size
	void modify_gaussians(gsplat::data::database* db, std::span<const float> grad, float scene_extent, std::mt19937& rng);
}

#endif // GSPLAT_TRAIN_MODIFY_H
