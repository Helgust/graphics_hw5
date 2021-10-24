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
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}
  };

  // Создание и аллокация буферов
  m_A = vk_utils::createBuffer(m_device, sizeof(float) * m_length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_B = vk_utils::createBuffer(m_device, sizeof(float) * m_length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  m_sum = vk_utils::createBuffer(m_device, sizeof(float) * m_length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  vk_utils::allocateAndBindWithPadding(m_device, m_physicalDevice, {m_A, m_B, m_sum}, 0);

  m_pBindings = std::make_unique<vk_utils::DescriptorMaker>(m_device, dtypes, 2);
  
  // Создание descriptor set для передачи буферов в шейдер
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, m_A);
  m_pBindings->BindBuffer(1, m_B);
  m_pBindings->BindEnd(&m_sumDS.emplace_back(), &m_sumDSLayout);
  
    // Создание descriptor set для передачи буферов в шейдер
  m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pBindings->BindBuffer(0, m_B);
  m_pBindings->BindBuffer(1, m_sum);
  m_pBindings->BindEnd(&m_sumDS.emplace_back(), &m_sumDSLayout);
  
  // Заполнение буферов
  std::vector<float> values(m_length);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i;
  }
  m_pCopyHelper->UpdateBuffer(m_A, 0, values.data(), sizeof(float) * values.size());
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i * i;
  }
  m_pCopyHelper->UpdateBuffer(m_A, 0, values.data(), sizeof(float) * values.size());
}

void SimpleCompute::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  // Заполняем буфер команд
  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[0]);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_sumDS[0], 0, NULL);

  vkCmdPushConstants(a_cmdBuff, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_length), &m_length);
  
  vkCmdDispatch(a_cmdBuff, m_length/BLOCK_SIZE, 1, 1);

  VkBufferMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.buffer = m_B;
  barrier.offset = 0;
  barrier.size = m_length*sizeof(float);

  vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,{},0, nullptr,1, &barrier,0, nullptr);

  vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[1]);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_sumDS[1], 0, NULL);

  vkCmdPushConstants(a_cmdBuff, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_length), &m_length);
  
  vkCmdDispatch(a_cmdBuff, m_length/BLOCK_SIZE, 1, 1);

  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.buffer = m_B;
  barrier.offset = 0;
  barrier.size = m_length*sizeof(float);
  vkCmdPipelineBarrier(a_cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,{},0, nullptr,1, &barrier,0, nullptr);


  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}


void SimpleCompute::CleanupPipeline()
{
  if (m_cmdBufferCompute)
  {
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_cmdBufferCompute);
  }

  vkDestroyBuffer(m_device, m_A, nullptr);
  vkDestroyBuffer(m_device, m_B, nullptr);
  vkDestroyBuffer(m_device, m_sum, nullptr);

  vkDestroyPipelineLayout(m_device, m_layout, nullptr);
  for (auto pipeline : m_pipelines)
  {
    vkDestroyPipeline(m_device, pipeline, nullptr);
  }
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

  // Загружаем шейдер
  std::vector<uint32_t> code1 = vk_utils::readSPVFile("../resources/shaders/scan1.comp.spv");
  VkShaderModuleCreateInfo createInfo1 = {};
  createInfo1.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo1.pCode    = code1.data();
  createInfo1.codeSize = code1.size()*sizeof(uint32_t);
    
  VkShaderModule shaderModule1;
  // Создаём шейдер в вулкане

  VK_CHECK_RESULT(vkCreateShaderModule(m_device, &createInfo1, NULL, &shaderModule1));

    std::vector<uint32_t> code2 = vk_utils::readSPVFile("../resources/shaders/scan2.comp.spv");
  VkShaderModuleCreateInfo createInfo2 = {};
  createInfo2.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo2.pCode    = code2.data();
  createInfo2.codeSize = code2.size()*sizeof(uint32_t);
    
  VkShaderModule shaderModule2;
  
  // Создаём шейдер в вулкане
  VK_CHECK_RESULT(vkCreateShaderModule(m_device, &createInfo2, NULL, &shaderModule2));

  std::vector<VkComputePipelineCreateInfo> pipelineCreateInfos = {};

  VkPipelineShaderStageCreateInfo shaderStageCreateInfo1 = {};
  shaderStageCreateInfo1.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo1.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo1.module = shaderModule1;
  shaderStageCreateInfo1.pName  = "main";

   VkPipelineShaderStageCreateInfo shaderStageCreateInfo2 = {};
  shaderStageCreateInfo2.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo2.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo2.module = shaderModule2;
  shaderStageCreateInfo2.pName  = "main";

  VkPushConstantRange pcRange = {};
  pcRange.offset = 0;
  pcRange.size = sizeof(m_length);
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // Создаём layout для pipeline
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts    = &m_sumDSLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges = &pcRange;
  VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, NULL, &m_layout));

  VkComputePipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stage  = shaderStageCreateInfo1;
  pipelineCreateInfo.layout = m_layout;

  pipelineCreateInfos.push_back(pipelineCreateInfo);

  pipelineCreateInfo.stage  = shaderStageCreateInfo2;

  pipelineCreateInfos.push_back(pipelineCreateInfo);
  
  // Создаём pipeline - объект, который выставляет шейдер и его параметры
  m_pipelines.resize(pipelineCreateInfos.size());
  VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, pipelineCreateInfos.size(), pipelineCreateInfos.data(), NULL, m_pipelines.data()));
  vkDestroyShaderModule(m_device, shaderModule1, nullptr);
  vkDestroyShaderModule(m_device, shaderModule2, nullptr);
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
  m_pCopyHelper->ReadBuffer(m_sum, 0, values.data(), sizeof(float) * values.size());
  for (auto v: values) {
    std::cout << v << ' ';
  }
}
