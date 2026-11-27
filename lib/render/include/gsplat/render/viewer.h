#ifndef GSPLAT_RENDER_VIEWER_H
#define GSPLAT_RENDER_VIEWER_H

#include <gsplat/train/scene.h>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>

struct GLFWwindow;

namespace gsplat::render {
	struct viewer_settings {
		std::string title = "gsplat";
		int width = 1280;
		int height = 720;
	};

	// a window that shows a scene from a camera you move with the mouse:
	// left drag orbits, right drag pans, scroll zooms. it owns the window,
	// swapchain and imgui, and borrows the vulkan handles and the scene.
	// the queue has to do graphics and present to the window, and the
	// instance and device need the extensions glfw asks for + VK_KHR_swapchain
	class viewer {
		// where the orbit camera is, around target, with up as the turntable axis
		struct orbit_t {
			glm::vec3 target{0, 0, 0};
			glm::vec3 up{0, -1, 0};      // COLMAP worlds tend to have y down
			glm::vec3 forward{0, 0, 1};  // which way yaw = 0 looks, perpendicular to up
			float yaw = 0;
			float pitch = 0;
			float distance = 5;
			float fovy = glm::radians(60.0f);
		};

		gsplat::train::vulkan_handles vk_v;
		gsplat::train::scene* scene_v;

		GLFWwindow* window_v = nullptr;
		VkSurfaceKHR surface_v = VK_NULL_HANDLE;
		VkSwapchainKHR swapchain_v = VK_NULL_HANDLE;
		VkFormat swapchain_format_v = VK_FORMAT_UNDEFINED;
		VkColorSpaceKHR color_space_v = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		VkExtent2D extent_v{0, 0};
		glm::ivec2 fb_size_v{0, 0}; // window size the swapchain was made for
		std::vector<VkImageView> swapchain_views_v;
		std::vector<VkFramebuffer> framebuffers_v;
		std::vector<VkSemaphore> render_finished_v; // one per swapchain image
		bool swapchain_dirty_v = false;
		VkRenderPass render_pass_v = VK_NULL_HANDLE;
		VkCommandPool command_pool_v = VK_NULL_HANDLE;
		VkCommandBuffer command_buffer_v = VK_NULL_HANDLE;
		VkSemaphore image_available_v = VK_NULL_HANDLE;
		VkFence in_flight_v = VK_NULL_HANDLE;
		bool imgui_v = false;

		// the scene's image as an imgui texture, remade when the scene remakes the image
		VkDescriptorSet texture_v = VK_NULL_HANDLE;
		VkImageView texture_view_v = VK_NULL_HANDLE;
		glm::ivec2 texture_size_v{0, 0};

		orbit_t orbit_v;
		orbit_t start_v;
		float render_scale_v = 0.5f;
		glm::vec3 background_v{0, 0, 0};
		float render_ms_v = 0;
		int render_width_v = 0;
		int render_height_v = 0;

		void create_swapchain();
		void destroy_swapchain();
		void update_orbit();
		void draw_panel();
		glm::mat4 view_matrix() const;
		void release();
	public:
		viewer(const gsplat::train::vulkan_handles& vk, gsplat::train::scene* scene, const viewer_settings& settings = {});
		~viewer();

		viewer(const viewer&) = delete;
		viewer(viewer&&) = delete;
		viewer& operator =(const viewer&) = delete;
		viewer& operator =(viewer&&) = delete;

		// orbits around the gaussians' center from outside them
		void look_at(std::span<const gsplat::data::gaussian> gaussians);
		// starts exactly where a dataset camera was (same pose and vertical fov),
		// orbiting the gaussians' center. image_height is the camera's image
		void look_from(const gsplat::train::camera& cam, int image_height, std::span<const gsplat::data::gaussian> gaussians);

		// draws one frame. false once the window is closed. the scene can take
		// new gaussians between frames, which is how training shows up live
		bool frame();
		// frame() until the window is closed
		void run();
	};
}

#endif // GSPLAT_RENDER_VIEWER_H
