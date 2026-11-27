#define GLFW_INCLUDE_NONE // no GL headers, this is vulkan only
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <gsplat/render/viewer.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

using namespace gsplat::render;
using gsplat::train::camera;
using gsplat::train::rendered_image;
using gsplat::data::gaussian;

static void check(VkResult r, const char* what) {
	if (r != VK_SUCCESS) throw std::runtime_error(std::string{"viewer: "} + what + " failed (VkResult " + std::to_string(r) + ")");
}

// per axis median of the means, so a few far off COLMAP points can't drag
// it, and the median distance to it
static void center_and_spread(std::span<const gaussian> gaussians, glm::vec3& center, float& spread) {
	center = glm::vec3(0);
	spread = 1;
	if (gaussians.empty()) return;

	std::vector<float> v(gaussians.size());
	auto mid = v.begin() + v.size() / 2;
	for (int axis = 0; axis < 3; axis++) {
		for (std::size_t i = 0; i < v.size(); i++) v[i] = gaussians[i].mean[axis];
		std::nth_element(v.begin(), mid, v.end());
		center[axis] = *mid;
	}
	for (std::size_t i = 0; i < v.size(); i++) v[i] = glm::length(glm::vec3(gaussians[i].mean) - center);
	std::nth_element(v.begin(), mid, v.end());
	spread = std::max(*mid, 1e-3f);
}

viewer::viewer(const gsplat::train::vulkan_handles& vk, gsplat::train::scene* scene, const viewer_settings& settings) : vk_v(vk), scene_v(scene) {
	if (!scene_v) throw std::runtime_error("viewer: no scene");
	if (!vk.instance || !vk.physical_device || !vk.device || !vk.queue) throw std::runtime_error("viewer: vulkan_handles is missing something");

	try {
		// what the handles have to support, checked up front so a missing
		// extension says so instead of crashing somewhere in the driver
		std::uint32_t family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(vk.physical_device, &family_count, nullptr);
		std::vector<VkQueueFamilyProperties> families(family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(vk.physical_device, &family_count, families.data());
		if (vk.queue_family_index >= family_count || !(families[vk.queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			throw std::runtime_error("viewer: the queue family can't do graphics");
		}
		if (!vkGetDeviceProcAddr(vk.device, "vkCreateSwapchainKHR")) {
			throw std::runtime_error("viewer: the VkDevice was made without VK_KHR_swapchain");
		}
		// imgui draws the scene's image through a linear sampler
		VkFormatProperties format_props;
		vkGetPhysicalDeviceFormatProperties(vk.physical_device, VK_FORMAT_R32G32B32A32_SFLOAT, &format_props);
		if (!(format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("viewer: this GPU can't filter VK_FORMAT_R32G32B32A32_SFLOAT images");
		}

		if (!glfwInit()) throw std::runtime_error("viewer: glfwInit failed");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window_v = glfwCreateWindow(settings.width, settings.height, settings.title.c_str(), nullptr, nullptr);
		if (!window_v) throw std::runtime_error("viewer: glfwCreateWindow failed");

		VkResult r = glfwCreateWindowSurface(vk.instance, window_v, nullptr, &surface_v);
		if (r == VK_ERROR_EXTENSION_NOT_PRESENT) {
			throw std::runtime_error("viewer: the VkInstance was made without the extensions from glfwGetRequiredInstanceExtensions()");
		}
		check(r, "glfwCreateWindowSurface");
		VkBool32 can_present = VK_FALSE;
		check(vkGetPhysicalDeviceSurfaceSupportKHR(vk.physical_device, vk.queue_family_index, surface_v, &can_present), "vkGetPhysicalDeviceSurfaceSupportKHR");
		if (!can_present) throw std::runtime_error("viewer: the queue can't present to this window");

		// UNORM, not SRGB: the render's values are already display values
		// (training fits them to the photos' bytes / 255), and imgui assumes it too
		std::uint32_t format_count = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface_v, &format_count, nullptr);
		std::vector<VkSurfaceFormatKHR> formats(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface_v, &format_count, formats.data());
		if (formats.empty()) throw std::runtime_error("viewer: the window has no surface formats");
		VkSurfaceFormatKHR chosen = formats[0];
		for (auto& f : formats) {
			if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				chosen = f;
				break;
			}
		}
		swapchain_format_v = chosen.format;
		color_space_v = chosen.colorSpace;

		VkAttachmentDescription color{};
		color.format = swapchain_format_v;
		color.samples = VK_SAMPLE_COUNT_1_BIT;
		color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_ref;
		// the attachment write waits for the acquire semaphore, which is waited at this stage
		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		VkRenderPassCreateInfo pass_info{};
		pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		pass_info.attachmentCount = 1;
		pass_info.pAttachments = &color;
		pass_info.subpassCount = 1;
		pass_info.pSubpasses = &subpass;
		pass_info.dependencyCount = 1;
		pass_info.pDependencies = &dependency;
		check(vkCreateRenderPass(vk.device, &pass_info, nullptr, &render_pass_v), "vkCreateRenderPass");

		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool_info.queueFamilyIndex = vk.queue_family_index;
		check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &command_pool_v), "vkCreateCommandPool");
		VkCommandBufferAllocateInfo command_buffer_info{};
		command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_info.commandPool = command_pool_v;
		command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_info.commandBufferCount = 1;
		check(vkAllocateCommandBuffers(vk.device, &command_buffer_info, &command_buffer_v), "vkAllocateCommandBuffers");

		VkSemaphoreCreateInfo semaphore_info{};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		check(vkCreateSemaphore(vk.device, &semaphore_info, nullptr, &image_available_v), "vkCreateSemaphore");
		VkFenceCreateInfo fence_info{};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // the first frame has nothing to wait for
		check(vkCreateFence(vk.device, &fence_info, nullptr, &in_flight_v), "vkCreateFence");

		create_swapchain();

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		imgui_v = true;
		ImGui::GetIO().IniFilename = nullptr; // no imgui.ini next to the binary
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForVulkan(window_v, true);

		ImGui_ImplVulkan_InitInfo info = {};
		info.ApiVersion = VK_API_VERSION_1_1; // only picks how imgui loads dynamic rendering, which this doesn't use
		info.Instance = vk.instance;
		info.PhysicalDevice = vk.physical_device;
		info.Device = vk.device;
		info.QueueFamily = vk.queue_family_index;
		info.Queue = vk.queue;
		info.DescriptorPoolSize = 16; // imgui makes its own pool: the font atlas + the scene's image
		info.MinImageCount = 2;
		info.ImageCount = std::max<std::uint32_t>(2, (std::uint32_t)swapchain_views_v.size());
		info.PipelineInfoMain.RenderPass = render_pass_v;
		info.PipelineInfoMain.Subpass = 0;
		info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		info.CheckVkResultFn = [](VkResult err) {
			if (err < 0) throw std::runtime_error("viewer: imgui's vulkan backend got VkResult " + std::to_string(err));
		};
		ImGui_ImplVulkan_Init(&info);
	}
	catch (...) {
		release();
		throw;
	}
}

viewer::~viewer() {
	release();
}

void viewer::release() {
	if (vk_v.queue) vkQueueWaitIdle(vk_v.queue);

	if (imgui_v) {
		ImGuiIO& io = ImGui::GetIO();
		if (io.BackendRendererUserData) {
			if (texture_v) ImGui_ImplVulkan_RemoveTexture(texture_v);
			ImGui_ImplVulkan_Shutdown();
		}
		if (io.BackendPlatformUserData) ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		imgui_v = false;
	}
	texture_v = VK_NULL_HANDLE;
	texture_view_v = VK_NULL_HANDLE;

	destroy_swapchain();
	if (swapchain_v) vkDestroySwapchainKHR(vk_v.device, swapchain_v, nullptr);
	if (in_flight_v) vkDestroyFence(vk_v.device, in_flight_v, nullptr);
	if (image_available_v) vkDestroySemaphore(vk_v.device, image_available_v, nullptr);
	if (command_pool_v) vkDestroyCommandPool(vk_v.device, command_pool_v, nullptr);
	if (render_pass_v) vkDestroyRenderPass(vk_v.device, render_pass_v, nullptr);
	if (surface_v) vkDestroySurfaceKHR(vk_v.instance, surface_v, nullptr);
	if (window_v) glfwDestroyWindow(window_v);
	glfwTerminate(); // does nothing if glfwInit never ran
	swapchain_v = VK_NULL_HANDLE;
	in_flight_v = VK_NULL_HANDLE;
	image_available_v = VK_NULL_HANDLE;
	command_pool_v = VK_NULL_HANDLE;
	render_pass_v = VK_NULL_HANDLE;
	surface_v = VK_NULL_HANDLE;
	window_v = nullptr;
}

// everything that hangs off the swapchain images, but not the swapchain
// itself, so create_swapchain can hand the old one over as oldSwapchain
void viewer::destroy_swapchain() {
	for (VkFramebuffer f : framebuffers_v) vkDestroyFramebuffer(vk_v.device, f, nullptr);
	for (VkImageView v : swapchain_views_v) vkDestroyImageView(vk_v.device, v, nullptr);
	for (VkSemaphore s : render_finished_v) vkDestroySemaphore(vk_v.device, s, nullptr);
	framebuffers_v.clear();
	swapchain_views_v.clear();
	render_finished_v.clear();
}

void viewer::create_swapchain() {
	check(vkQueueWaitIdle(vk_v.queue), "vkQueueWaitIdle");
	destroy_swapchain();

	VkSurfaceCapabilitiesKHR caps;
	check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_v.physical_device, surface_v, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
	int fb_width = 0, fb_height = 0;
	glfwGetFramebufferSize(window_v, &fb_width, &fb_height);
	VkExtent2D extent = caps.currentExtent;
	if (extent.width == UINT32_MAX) {
		// the surface takes whatever size the swapchain says
		extent.width = std::clamp((std::uint32_t)fb_width, caps.minImageExtent.width, caps.maxImageExtent.width);
		extent.height = std::clamp((std::uint32_t)fb_height, caps.minImageExtent.height, caps.maxImageExtent.height);
	}
	std::uint32_t image_count = caps.minImageCount + 1;
	if (caps.maxImageCount != 0) image_count = std::min(image_count, caps.maxImageCount);
	VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	for (VkCompositeAlphaFlagBitsKHR c : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
		if (caps.supportedCompositeAlpha & c) {
			composite = c;
			break;
		}
	}

	VkSwapchainCreateInfoKHR info{};
	info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	info.surface = surface_v;
	info.minImageCount = image_count;
	info.imageFormat = swapchain_format_v;
	info.imageColorSpace = color_space_v;
	info.imageExtent = extent;
	info.imageArrayLayers = 1;
	info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.preTransform = caps.currentTransform;
	info.compositeAlpha = composite;
	info.presentMode = VK_PRESENT_MODE_FIFO_KHR; // vsync, and the one mode every driver has
	info.clipped = VK_TRUE;
	info.oldSwapchain = swapchain_v;
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	check(vkCreateSwapchainKHR(vk_v.device, &info, nullptr, &swapchain), "vkCreateSwapchainKHR");
	if (swapchain_v) vkDestroySwapchainKHR(vk_v.device, swapchain_v, nullptr);
	swapchain_v = swapchain;
	extent_v = extent;
	fb_size_v = glm::ivec2(fb_width, fb_height);
	swapchain_dirty_v = false;

	std::uint32_t count = 0;
	vkGetSwapchainImagesKHR(vk_v.device, swapchain_v, &count, nullptr);
	std::vector<VkImage> images(count);
	vkGetSwapchainImagesKHR(vk_v.device, swapchain_v, &count, images.data());
	for (VkImage image : images) {
		VkImageViewCreateInfo view_info{};
		view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_info.image = image;
		view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_info.format = swapchain_format_v;
		view_info.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		VkImageView view = VK_NULL_HANDLE;
		check(vkCreateImageView(vk_v.device, &view_info, nullptr, &view), "vkCreateImageView");
		swapchain_views_v.push_back(view);

		VkFramebufferCreateInfo fb_info{};
		fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fb_info.renderPass = render_pass_v;
		fb_info.attachmentCount = 1;
		fb_info.pAttachments = &view;
		fb_info.width = extent.width;
		fb_info.height = extent.height;
		fb_info.layers = 1;
		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		check(vkCreateFramebuffer(vk_v.device, &fb_info, nullptr, &framebuffer), "vkCreateFramebuffer");
		framebuffers_v.push_back(framebuffer);

		VkSemaphoreCreateInfo semaphore_info{};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkSemaphore semaphore = VK_NULL_HANDLE;
		check(vkCreateSemaphore(vk_v.device, &semaphore_info, nullptr, &semaphore), "vkCreateSemaphore");
		render_finished_v.push_back(semaphore);
	}
}

void viewer::look_at(std::span<const gaussian> gaussians) {
	glm::vec3 center;
	float spread;
	center_and_spread(gaussians, center, spread);
	float fovy = orbit_v.fovy;
	orbit_v = orbit_t{};
	orbit_v.target = center;
	orbit_v.distance = 3 * spread;
	orbit_v.fovy = fovy;
	start_v = orbit_v;
}

void viewer::look_from(const camera& cam, int image_height, std::span<const gaussian> gaussians) {
	// world_to_camera is OpenCV (x right, y down, z forward), so the camera's
	// axes in the world are the rows of its rotation
	glm::mat3 R_t = glm::transpose(glm::mat3(cam.world_to_camera));
	glm::vec3 eye = -(R_t * glm::vec3(cam.world_to_camera[3]));
	glm::vec3 forward = R_t * glm::vec3(0, 0, 1);
	glm::vec3 up = -(R_t * glm::vec3(0, 1, 0));

	glm::vec3 center;
	float spread;
	center_and_spread(gaussians, center, spread);
	// orbit the point straight ahead that's as far as the gaussians' center
	float d = glm::dot(center - eye, forward);
	if (!(d > 1e-3f)) d = 3 * spread; // center is behind the camera

	orbit_v = orbit_t{};
	orbit_v.target = eye + d * forward;
	orbit_v.up = up;
	orbit_v.forward = forward;
	orbit_v.distance = d;
	orbit_v.fovy = 2 * std::atan(0.5f * (float)image_height / cam.intrinsics.y);
	start_v = orbit_v;
}

glm::mat4 viewer::view_matrix() const {
	// yaw turns around up, pitch tilts toward it. yaw = pitch = 0 is looking along forward
	const orbit_t& o = orbit_v;
	glm::vec3 right = glm::normalize(glm::cross(o.forward, o.up));
	glm::vec3 to_eye = std::cos(o.pitch) * (std::cos(o.yaw) * -o.forward + std::sin(o.yaw) * right) + std::sin(o.pitch) * o.up;
	return glm::lookAt(o.target + o.distance * to_eye, o.target, o.up);
}

void viewer::update_orbit() {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return; // the mouse is on the panel

	ImVec2 d = io.MouseDelta;
	if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
		// turntable: the scene follows the mouse
		orbit_v.yaw -= 0.005f * d.x;
		orbit_v.pitch = std::clamp(orbit_v.pitch + 0.005f * d.y, -1.5f, 1.5f);
	}
	if (ImGui::IsMouseDown(ImGuiMouseButton_Right) || ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
		// moves the target so the point under the mouse stays under it
		glm::mat4 view = view_matrix();
		glm::vec3 cam_right{view[0][0], view[1][0], view[2][0]};
		glm::vec3 cam_up{view[0][1], view[1][1], view[2][1]};
		float per_pixel = 2 * orbit_v.distance * std::tan(0.5f * orbit_v.fovy) / std::max(io.DisplaySize.y, 1.0f);
		orbit_v.target += per_pixel * (-d.x * cam_right + d.y * cam_up);
	}
	if (io.MouseWheel != 0) {
		orbit_v.distance *= std::pow(0.9f, io.MouseWheel);
	}
}

void viewer::draw_panel() {
	ImGuiIO& io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowBgAlpha(0.8f);
	ImGui::Begin("view", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("%zu gaussians", scene_v->gaussian_count());
	ImGui::Text("%dx%d render, %.1f ms", render_width_v, render_height_v, render_ms_v);
	ImGui::Text("%.0f fps", io.Framerate);
	ImGui::SliderFloat("render scale", &render_scale_v, 0.1f, 1.0f, "%.2f");
	float fov = glm::degrees(orbit_v.fovy);
	if (ImGui::SliderFloat("fov", &fov, 10.0f, 120.0f, "%.0f deg")) orbit_v.fovy = glm::radians(fov);
	ImGui::ColorEdit3("background", &background_v.x);
	if (ImGui::Button("reset view")) orbit_v = start_v;
	ImGui::TextDisabled("left drag orbit, right drag pan, scroll zoom");
	ImGui::End();
}

bool viewer::frame() {
	glfwPollEvents();
	if (glfwWindowShouldClose(window_v)) return false;

	int fb_width = 0, fb_height = 0;
	glfwGetFramebufferSize(window_v, &fb_width, &fb_height);
	if (fb_width == 0 || fb_height == 0) {
		glfwWaitEvents(); // minimized
		return true;
	}
	if (swapchain_dirty_v || fb_size_v != glm::ivec2(fb_width, fb_height)) create_swapchain();

	// last frame has to be off the GPU before the scene overwrites its image
	check(vkWaitForFences(vk_v.device, 1, &in_flight_v, VK_TRUE, UINT64_MAX), "vkWaitForFences");
	std::uint32_t image_index = 0;
	VkResult acquired = vkAcquireNextImageKHR(vk_v.device, swapchain_v, UINT64_MAX, image_available_v, VK_NULL_HANDLE, &image_index);
	if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
		swapchain_dirty_v = true;
		return true;
	}
	if (acquired == VK_SUBOPTIMAL_KHR) swapchain_dirty_v = true; // still got an image, remake after this frame
	else check(acquired, "vkAcquireNextImageKHR");
	check(vkResetFences(vk_v.device, 1, &in_flight_v), "vkResetFences");

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	update_orbit();

	render_width_v = std::max(1, (int)std::lround((float)fb_width * render_scale_v));
	render_height_v = std::max(1, (int)std::lround((float)fb_height * render_scale_v));
	// near scales with the zoom, so closing in on something doesn't clip it
	glm::mat4 projection = glm::perspective(orbit_v.fovy, (float)render_width_v / (float)render_height_v, 0.01f * orbit_v.distance, 1000.0f * orbit_v.distance);
	auto t0 = std::chrono::steady_clock::now();
	rendered_image img = scene_v->render(view_matrix(), projection, render_width_v, render_height_v, glm::vec4(background_v, 0));
	float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
	render_ms_v = render_ms_v == 0 ? ms : 0.9f * render_ms_v + 0.1f * ms;

	// the scene remakes its image when the size changes, and a new view can
	// reuse the old handle, so the size is part of the check
	if (img.view != texture_view_v || glm::ivec2(img.width, img.height) != texture_size_v) {
		if (texture_v) ImGui_ImplVulkan_RemoveTexture(texture_v);
		texture_v = ImGui_ImplVulkan_AddTexture(img.view, img.layout);
		texture_view_v = img.view;
		texture_size_v = glm::ivec2(img.width, img.height);
	}
	ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)texture_v, ImVec2(0, 0), ImGui::GetIO().DisplaySize);
	draw_panel();
	ImGui::Render();

	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	check(vkBeginCommandBuffer(command_buffer_v, &begin), "vkBeginCommandBuffer");
	VkClearValue clear{};
	VkRenderPassBeginInfo pass{};
	pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	pass.renderPass = render_pass_v;
	pass.framebuffer = framebuffers_v[image_index];
	pass.renderArea.extent = extent_v;
	pass.clearValueCount = 1;
	pass.pClearValues = &clear;
	vkCmdBeginRenderPass(command_buffer_v, &pass, VK_SUBPASS_CONTENTS_INLINE);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer_v);
	vkCmdEndRenderPass(command_buffer_v);
	check(vkEndCommandBuffer(command_buffer_v), "vkEndCommandBuffer");

	VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &image_available_v;
	submit.pWaitDstStageMask = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &command_buffer_v;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &render_finished_v[image_index];
	check(vkQueueSubmit(vk_v.queue, 1, &submit, in_flight_v), "vkQueueSubmit");

	VkPresentInfoKHR present{};
	present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present.waitSemaphoreCount = 1;
	present.pWaitSemaphores = &render_finished_v[image_index];
	present.swapchainCount = 1;
	present.pSwapchains = &swapchain_v;
	present.pImageIndices = &image_index;
	VkResult presented = vkQueuePresentKHR(vk_v.queue, &present);
	if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) swapchain_dirty_v = true;
	else check(presented, "vkQueuePresentKHR");
	return true;
}

void viewer::run() {
	while (frame()) {
	}
}
