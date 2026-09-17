#include <gsplat/vulkan/physical_device.h>

using namespace gsplat::vulkan;

phyisical_device::~phyisical_device() {

}
VkPhysicalDevice phyisical_device::device_handle() {
	return device_v;
}
VkInstance phyisical_device::instance_handle() {
	return instance_ref_v;
}