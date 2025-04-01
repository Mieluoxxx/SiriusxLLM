/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-24 15:24:33
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-04-01 13:12:31
 * @FilePath: /SiriusxLLM/siriusx/src/model/model.cpp
 * @Description: 审查完成 0228
 */
#include "model/model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>

#include "base/base.h"
#include "base/buffer.h"
#include "op/encode.h"

// clang-format off
 namespace model {
 // 构造函数，用于初始化Model对象
 Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
              std::string token_path, std::string model_path, bool is_quant_model)
     : tokenizer_type_(tokenizer_type),         // 初始化tokenizer_type_成员变量
       model_type_(model_type),                 // 初始化model_type_成员变量
       token_path_(std::move(token_path)),      // 初始化token_path_成员变量，使用std::move将token_path参数移动到token_path_成员变量中
       model_path_(std::move(model_path)),      // 初始化model_path_成员变量，使用std::move将model_path参数移动到model_path_成员变量中
       is_quant_model_(is_quant_model) {}       // 初始化is_quant_model_成员变量
 
 // 返回模型类型
 base::ModelType Model::model_type() const { return model_type_; }
 
 // 返回token_path_的引用
 const std::string& Model::token_path() const { return token_path_; }
 
 // 返回模型路径的引用
 const std::string& Model::model_path() const { return model_path_; }
 
 // 向模型中插入缓冲区
 base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
     // 如果缓冲区中已经存在该索引，则返回错误
     if (buffers_.count(buffer_idx) > 0) {
         return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffer");
     }
     // 如果要插入的tensor为空，则返回错误
     if (tensor.is_empty()) {
         return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
     }
     // 将tensor插入到缓冲区中
     buffers_.insert({buffer_idx, tensor});
     // 返回成功
     return base::error::Success();
 }
 
 // 获取指定类型的缓冲区
 tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
     // 检查缓冲区是否存在
     CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
     // 返回指定类型的缓冲区
     return buffers_.at(buffer_idx);
 }
 
 // 获取指定类型的缓冲区
 const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
     // 检查缓冲区是否存在
     CHECK_GT(buffers_.count(buffer_idx), 0);
     // 返回指定类型的缓冲区
     return buffers_.at(buffer_idx);
 }
 
 base::Status Model::read_model_file() {
     // 使用命名空间base
     using namespace base;
     // 如果模型路径为空，则返回错误
     if (model_path_.empty()) {
       return error::PathNotValid("Failed to open the weight file, the model path is empty!");
     }
     // 打开模型文件，只读模式
     int32_t fd = open(model_path_.data(), O_RDONLY);
     // 如果打开失败，则返回错误
     if (fd == -1) {
       return error::PathNotValid("Failed to open the weight file " + model_path_ + " may be the path does not exist!");
     }
   
     // 以二进制读模式打开文件
     FILE* file = fopen(model_path_.data(), "rb");
     // 如果打开失败，则返回错误
     if (!file) {
       return error::PathNotValid("Failed to open the file. The path may be invalid.");
     }
   
     // 定义模型配置
     auto config = ModelConfig{};
     // 从文件中读取模型配置
     if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
       return error::ModelParseError(
           "Failed to retrieve the configuration information from the model file.");
     }
     // 如果是量化模型，则从文件中读取组大小信息
     if (is_quant_model_) {
       if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
         return error::ModelParseError(
             "Failed to retrieve the group size information from the model file.");
       }
     }
   
     auto gen_status = generate_model_infos(config);
     if (!gen_status) {
       return gen_status;
     }
   
     if (!is_quant_model_) {
       raw_model_data_ = std::make_shared<RawModelDataFP32>();
     } else {
       raw_model_data_ = std::make_shared<RawModelDataINT8>();
     }
   
       // 如果是量化模型，则创建RawModelDataINT8对象
     struct stat st;
     if (fstat(fd, &st) == -1) {
       close(fd);
       return error::ModelParseError("Failed to retrieve the file size information from the model file.");
     }
     raw_model_data_->file_size = st.st_size;
     LOG(INFO) << "The tokenizer model path: " << token_path_;
     std::string tokenizer_type_str = tokenizer_type_ == TokenizerType::EncodeBpe ? "Bpe" : "Spe";
     LOG(INFO) << "The tokenizer type: " << tokenizer_type_str;
   
     LOG(INFO) << "The model path: " << model_path_;
     LOG(INFO) << "The model file size: " << raw_model_data_->file_size << " byte";
     std::string quant_info = is_quant_model_ ? "quant" : "not quant";
     LOG(INFO) << "The model is " << quant_info << " model";
   
     if (config_) {
       LOG(INFO) << "\nThe model info: " << *config_;
     }
   
     raw_model_data_->fd = fd;
     raw_model_data_->data =
         mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
   
     if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
       return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
     }
     if (!is_quant_model_) {
       raw_model_data_->weight_data =
           static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
     } else {
       raw_model_data_->weight_data =
           static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
     }
     if (raw_model_data_ == nullptr) {
       LOG(ERROR);
       return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                     " into memory, the pointer to weight start address is null");
     }
     return error::Success();
 }
 
 // 生成模型信息
 base::Status Model::generate_model_infos(const ModelConfig& config) const {
     // 设置模型维度
     config_->dim_ = config.dim;
     // 设置隐藏层维度
     config_->hidden_dim_ = config.hidden_dim;
     // 设置层数
     config_->layer_num_ = config.layer_num;
     // 设置头数
     config_->head_num_ = config.head_num;
     // 设置键值头数
     config_->kv_head_num_ = config.kv_head_num;
     // 设置序列长度
     config_->seq_len_ = config.seq_len;
 
     // 计算键值维度
     config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
     // 计算键值乘数
     config_->kv_mul_ = config.head_num / config.kv_head_num;
     // 计算头大小
     config_->head_size_ = config.dim / config.head_num;
 
     // 如果词汇表大小大于0，则设置共享权重为true
     if (config.vocab_size > 0) {
         config_->is_shared_weight_ = true;
     } else {
         // 否则设置共享权重为false
         config_->is_shared_weight_ = false;
     }
 
     // 设置词汇表大小
     config_->vocab_size_ = std::abs(config.vocab_size);
     // 返回成功状态
     return base::error::Success();
 }
 
 // 创建编码层
 base::Status Model::create_encode_layer() {
     using namespace base;
 
     // 如果tokenizer_type_为EncodeSpe，则创建SpeEncodeLayer
     if(tokenizer_type_ == TokenizerType::EncodeSpe) {
         encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
     } else {
#ifdef LLAMA3_SUPPORT
        encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif
#ifdef QWEN2_SUPPORT
        encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
     }
 
     // 如果encode_layer_为空，则返回错误
     if(!encode_layer_) {
         return error::InternalError("The vocab size param read error from the model file!");
     }
     // 将encode_layer_的vocab_size_赋值给config_->vocab_size_
     config_->vocab_size_ = encode_layer_->vocab_size();
     // 如果vocab_size_小于等于0，则返回错误
     if (config_->vocab_size_ <= 0) {
         return error::InternalError("The vocab size param read error from the model file!");
       }
     // 返回成功
     return error::Success();
 }
 
base::Status Model::gen_model_from_file() {
     // 使用命名空间base
     using namespace base;
     // 创建TransformerConfig对象
     config_ = std::make_unique<TransformerConfig>();
     // 创建编码层
     auto create_encode_status = create_encode_layer();
     // 如果创建编码层失败，则返回错误信息
     if (!create_encode_status) {
         LOG(ERROR) << "Create the encode layer failed! " << create_encode_status.get_err_msg();
         return create_encode_status;
     }
 
     // mmap
     // 读取模型文件
     auto mmap_status = read_model_file();
     // 如果读取模型文件失败，则返回错误信息
     if (!mmap_status) {
       LOG(ERROR) << "Read model file " << model_path_ << " failed! " << mmap_status.get_err_msg();
       return mmap_status;
     }
 
     // 创建模型文件中的层
     auto layer_create_status = create_layers();
     // 如果创建层失败，则返回错误信息
     if (!layer_create_status) {
       LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed! "
                  << mmap_status.get_err_msg();
       return layer_create_status;
     }
     // 返回成功状态
     return error::Success();
 }
 
 
 // 定义一个名为Model的类，其中包含一个名为encode的成员函数，该函数接受一个字符串参数，返回一个int32_t类型的向量
 std::vector<int32_t> Model::encode(const std::string& sentence) const {
     // 检查encode_layer_是否为空，如果不为空，则调用encode_layer_的encode函数，将sentence作为参数传入，返回一个int32_t类型的向量
     CHECK(encode_layer_ != nullptr);
     return encode_layer_->encode(sentence);
 }
 
 // 判断给定token索引是否为句子结束
 bool Model::is_sentence_ending(int32_t token_idx) const {
     // 检查encode_layer_是否为空
     CHECK(this->encode_layer_ != nullptr);
     // 调用encode_layer_的is_sentence_ending方法，判断给定token索引是否为句子结束
     return this->encode_layer_->is_sentence_ending(token_idx);
 }
 
 // 根据给定的token索引解码
 std::string Model::decode(int32_t token_idx) const {
     // 检查encode_layer_是否为空
     CHECK(this->encode_layer_ != nullptr);
     // 调用encode_layer_的decode函数进行解码
     return this->encode_layer_->decode(token_idx);
 }
 
 // 根据给定的token索引解码模型
std::string Model::decode(std::vector<int32_t> token_idxs) const {
     // 检查编码层是否为空
     CHECK(this->encode_layer_ != nullptr);
     // 调用编码层的解码函数
     return this->encode_layer_->decode(token_idxs);
 }
 
 // 根据给定的层索引和token位置，从key-value缓存中切片出对应的key和value
 std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
     // 计算当前层的偏移量
     int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
     // 计算当前token在缓存中的偏移量
     int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
     
     // 获取key和value缓存的指针
     float* key_cache_ptr = const_cast<float*>(get_buffer(ModelBufferType::KeyCache).ptr<float>(cache_offset));
     float* value_cache_ptr = const_cast<float*>(get_buffer(ModelBufferType::ValueCache).ptr<float>(cache_offset));
 
     // 创建key和value的tensor
     tensor::Tensor key(base::DataType::FP32, config_->kv_dim_, false, nullptr, key_cache_ptr);
     tensor::Tensor value(base::DataType::FP32, config_->kv_dim_, false, nullptr, value_cache_ptr);
 
     // 设置tensor的设备类型
     key.set_device_type(device_type_);
     value.set_device_type(device_type_);
     // 返回key和value的tensor
     return {key, value};
 }
 
 // 根据位置张量和嵌入输出填充输入张量
 tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor, const op::EmbeddingOutput& embedding_output, bool is_prompt) const {
     // 获取位置张量的第一个元素
     const int32_t pos = pos_tensor.index<int32_t>(0);
     // 获取嵌入输出的三个元素：输入标记、输入嵌入和输入标记数量
     auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
     
     // 初始化索引
     int32_t index = 0;
     // 如果是提示，则索引为位置
     if (is_prompt) index = pos;
     // 创建输入嵌入缓冲区，大小为维度乘以浮点数大小，初始值为输入嵌入的指针，是否为共享指针为true
     std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr, input_embeddings.ptr<float>(index * config_->dim_), true);
 
     // 创建输入张量，数据类型为浮点数，维度为配置中的维度
     tensor::Tensor input(base::DataType::FP32, config_->dim_);
     // 将输入嵌入缓冲区赋值给输入张量
     input.assign(input_emb_buffer);
     // 设置输入张量的设备类型
     input.set_device_type(device_type_);
     // 返回输入张量
     return input;
 }
 }  // namespace model