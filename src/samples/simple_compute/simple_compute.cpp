#include "simple_compute.h"

#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <vk_utils.h>

SimpleCompute::SimpleCompute(uint32_t a_length) : m_length(a_length)
{
#ifdef NDEBUG
  m_enableValidation = false;
#else
  m_enableValidation = true;
#endif
}


void SimpleCompute::SetupValidationLayers()
{
  m_validationLayers.push_back("VK_LAYER_KHRONOS_validation");
  m_validationLayers.push_back("VK_LAYER_LUNARG_monitor");
}

void SimpleCompute::InitVulkan(const char** a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId)
{
  m_instanceExtensions.clear();
  for (uint32_t i = 0; i < a_instanceExtensionsCount; ++i) {
    m_instanceExtensions.push_back(a_instanceExtensions[i]);
  }
  SetupValidationLayers();
  VK_CHECK_RESULT(volkInitialize());
  CreateInstance();
  volkLoadInstance(m_instance);

  CreateDevice(a_deviceId);
  volkLoadDevice(m_device);

  m_commandPool = vk_utils::createCommandPool(m_device, m_queueFamilyIDXs.compute, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  m_cmdBufferCompute = vk_utils::createCommandBuffers(m_device, m_commandPool, 1)[0];
  
  m_pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(m_physicalDevice, m_device, m_transferQueue, m_queueFamilyIDXs.compute, 8*1024*1024);
}


void SimpleCompute::CreateInstance()
{
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = nullptr;
  appInfo.pApplicationName = "VkRender";
  appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.pEngineName = "SimpleCompute";
  appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);

  m_instance = vk_utils::createInstance(m_enableValidation, m_validationLayers, m_instanceExtensions, &appInfo);
  if (m_enableValidation)
    vk_utils::initDebugReportCallback(m_instance, &debugReportCallbackFn, &m_debugReportCallback);
}

void SimpleCompute::CreateDevice(uint32_t a_deviceId)
{
  m_physicalDevice = vk_utils::findPhysicalDevice(m_instance, true, a_deviceId, m_deviceExtensions);

  m_device = vk_utils::createLogicalDevice(m_physicalDevice, m_validationLayers, m_deviceExtensions,
                                           m_enabledDeviceFeatures, m_queueFamilyIDXs,
                                           VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);

  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.compute, 0, &m_computeQueue);
  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.transfer, 0, &m_transferQueue);
}


void SimpleCompute::SetupSimplePipeline()
{
  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}
  };

  // Создание и аллокация буферов
  m_A = vk_utils::createBuffer(m_device, sizeof(float) * m_length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_G = vk_utils::createBuffer(m_device, sizeof(float) * GROUP_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  m_res = vk_utils::createBuffer(m_device, sizeof(float) * m_length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  vk_utils::allocateAndBindWithPadding(m_device, m_physicalDevice, {m_A, m_G, m_res}, 0);

  m_pBindings = std::make_unique<vk_utils::DescriptorMaker>(m_device, dtypes, 3);
  
  // Создание descriptor set для передачи буферов в шейдер
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, m_A);
  m_pBindings->BindBuffer(1, m_G);
  m_pBindings->BindBuffer(2, m_res);
  m_pBindings->BindEnd(&m_sumDS, &m_sumDSLayout);
  
    // Создание descriptor set для передачи буферов в шейдер
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, m_G);
  m_pBindings->BindBuffer(1, m_G);
  m_pBindings->BindBuffer(2, m_G);
  m_pBindings->BindEnd(&m_groupDS, &m_sumDSLayout);

  // Создание descriptor set для передачи буферов в шейдер
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, m_G);
  m_pBindings->BindBuffer(1, m_res);
  m_pBindings->BindEnd(&m_finalDS, &m_finalDSLayout);
  
  // Заполнение буферов
  std::vector<float> values(m_length);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 1.0f;
  }
  m_pCopyHelper->UpdateBuffer(m_A, 0, values.data(), sizeof(float) * values.size());

  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 0.0;
  }
  m_pCopyHelper->UpdateBuffer(m_G, 0, values.data(), sizeof(float) * GROUP_SIZE);
  m_pCopyHelper->UpdateBuffer(m_res, 0, values.data(), sizeof(float) * values.size());

}

void SimpleCompute::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline)
{

struct pushConst
{
  uint totalLength;
  uint depth;
};


  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  // Заполняем буфер команд
  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_sumPipeline);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_sumPipelineLayout, 0, 1, &m_sumDS, 0, NULL);
  pushConst val = pushConst{m_length, 0};
  vkCmdPushConstants(a_cmdBuff, m_sumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(val), &val);
  
  vkCmdDispatch(a_cmdBuff, (m_length/GROUP_SIZE) + (m_length % GROUP_SIZE != 0), 1, 1);

  VkBufferMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.buffer = m_G;
  barrier.offset = 0;
  barrier.size = GROUP_SIZE*sizeof(float);

  vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,{},0, nullptr,1, &barrier,0, nullptr);

  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_sumPipelineLayout, 0, 1, &m_groupDS, 0, NULL);

  pushConst val2 = pushConst{(m_length/GROUP_SIZE) + (m_length % GROUP_SIZE != 0), 1};
  vkCmdPushConstants(a_cmdBuff, m_sumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(val2), &val2);
  
  vkCmdDispatch(a_cmdBuff, 1, 1, 1);

  // barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  // barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  // barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  // barrier.buffer = m_B;
  // barrier.offset = 0;
  // barrier.size = m_length*sizeof(float);
  // vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,{},0, nullptr,1, &barrier,0, nullptr);

  vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_finalPipeline);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_finalPipelineLayout, 0, 1, &m_finalDS, 0, NULL);

  vkCmdPushConstants(a_cmdBuff, m_sumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_length), &m_length);
  
  vkCmdPipelineBarrier(a_cmdBuff,
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 0, nullptr, 1, &barrier, 0, nullptr);

  vkCmdDispatch(a_cmdBuff, (m_length / GROUP_SIZE) + (m_length % GROUP_SIZE != 0), 1, 1);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}


void SimpleCompute::CleanupPipeline()
{
  if (m_cmdBufferCompute)
  {
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_cmdBufferCompute);
  }

  vkDestroyBuffer(m_device, m_A, nullptr);
  vkDestroyBuffer(m_device, m_G, nullptr);
  vkDestroyBuffer(m_device, m_res, nullptr);

  vkDestroyPipelineLayout(m_device, m_sumPipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_sumPipeline, nullptr);

  vkDestroyPipelineLayout(m_device, m_finalPipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_finalPipeline, nullptr);

}


void SimpleCompute::Cleanup()
{
  CleanupPipeline();

  if (m_commandPool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
  }
}


void SimpleCompute::CreateComputePipeline()
{
auto create = [this] (const char *shader_path, uint32_t push_const_size, VkDescriptorSetLayout& DSLayout,
                         VkPipelineLayout& layout, VkPipeline& pipeline) {
      // Загружаем шейдер
      std::vector<uint32_t> code = vk_utils::readSPVFile(shader_path);
      VkShaderModuleCreateInfo createInfo = {};
      createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.pCode    = code.data();
      createInfo.codeSize = code.size()*sizeof(uint32_t);

      VkShaderModule shaderModule;
      // Создаём шейдер в вулкане
      VK_CHECK_RESULT(vkCreateShaderModule(m_device, &createInfo, NULL, &shaderModule));

      VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
      shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
      shaderStageCreateInfo.module = shaderModule;
      shaderStageCreateInfo.pName  = "main";

      VkPushConstantRange pcRange = {};
      pcRange.offset = 0;
      pcRange.size = push_const_size;
      pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      // Создаём layout для pipeline
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
      pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutCreateInfo.setLayoutCount = 1;
      pipelineLayoutCreateInfo.pSetLayouts    = &DSLayout;
      pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
      pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
      VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, NULL, &layout));

      VkComputePipelineCreateInfo pipelineCreateInfo = {};
      pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipelineCreateInfo.stage  = shaderStageCreateInfo;
      pipelineCreateInfo.layout = layout;

      // Создаём pipeline - объект, который выставляет шейдер и его параметры
      VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));

      vkDestroyShaderModule(m_device, shaderModule, nullptr);
  };

  create("../resources/shaders/scan1.comp.spv", 2*sizeof(m_length),
         m_sumDSLayout, m_sumPipelineLayout, m_sumPipeline);
  create("../resources/shaders/scan2.comp.spv", sizeof(m_length),
         m_finalDSLayout, m_finalPipelineLayout, m_finalPipeline);
}


void SimpleCompute::Execute()
{
  SetupSimplePipeline();
  
  CreateComputePipeline();

  BuildCommandBufferSimple(m_cmdBufferCompute, nullptr);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_cmdBufferCompute;

  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  VK_CHECK_RESULT(vkCreateFence(m_device, &fenceCreateInfo, NULL, &m_fence));

  // Отправляем буфер команд на выполнение
  VK_CHECK_RESULT(vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence));

  //Ждём конца выполнения команд
  VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, 100000000000));

  std::vector<float> values(m_length);
  m_pCopyHelper->ReadBuffer(m_res, 0, values.data(), sizeof(float) * values.size());
  for (auto v: values) {
    std::cout << v << ' ';
  }
}
