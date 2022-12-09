#ifndef __DUMMY_BACKEND_HPP
#define __DUMMY_BACKEND_HPP

#include <tmci/backend.hpp>
#include <json/json.h>
#include "client/client.hpp"
#include "client/utils.hpp"
namespace flamestore {

class MochiBackend : public tmci::Backend {

    Client*     m_client = nullptr;
    std::string m_model_name;
    std::string m_signature;

    public:

    MochiBackend(const char* config) {
        std::stringstream ss(config);
        Json::Value root;
        ss >> root;
        m_client = Client::from_id(root["flamestore_client"].asString());
        m_model_name = root["model_name"].asString();
        m_signature = root["signature"].asString();
    }

    ~MochiBackend() = default;

    virtual int Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
        // TODO check that m_client is valid
        //std::cout<<"enter save!!!!!!!\n";
	//fflush(stdout);
	std::vector<std::pair<void*,size_t>> segments;
        segments.reserve(tensors.size());
        std::vector<std::string> ptrs(tensors.size());
	size_t total_size = 0;
	if (isGPUPtr(tensors.begin()->get().tensor_data().data())){
		int iter=0;
        	for(auto &t: tensors){
            		auto size = (size_t)t.get().tensor_data().size();
            		total_size += size;
            		ptrs[iter].resize(size);
            		cudaMemcpy((char*)ptrs[iter].data(), (char*)t.get().tensor_data().data(), size, cudaMemcpyDeviceToHost);
            		segments.emplace_back((void*)ptrs[iter].data(), size);
            		iter++;
        	}
    	}else{
        	std::cout<<"not a GPU POINTER!\n";
		fflush(stdout);
        	for(const tensorflow::Tensor& t : tensors) {
            		total_size += t.tensor_data().size();
            		segments.emplace_back((void*)t.tensor_data().data(), (size_t)t.tensor_data().size());
        	}
	}
        Client::return_status status = m_client->write_model_data(m_model_name, m_signature, segments, total_size);
        return status.first;
   }
 
    virtual int Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
        // TODO check that m_client is valid
        std::vector<std::string> ptrs(tensors.size());
	std::vector<std::pair<void*,size_t>> segments;
        segments.reserve(tensors.size());
        size_t total_size = 0;
        
    	if (isGPUPtr(tensors.begin()->get().tensor_data().data())){
        	int iter=0;
        	for(auto &t: tensors){
            		size_t size = (size_t)t.get().tensor_data().size();
    	    		total_size += size;
            		ptrs[iter].resize(size);
            		segments.emplace_back((void*)ptrs[iter].data(), size);
            		iter++;
        	}
    	}else{
        	std::cout<<"NOT a GPU POINTER!\n";
		fflush(stdout);
        	for(auto &t: tensors){
            		segments.emplace_back((void*)t.get().tensor_data().data(), (size_t)t.get().tensor_data().size());
    			total_size +=  (size_t)t.get().tensor_data().size();
		}
    	}

        Client::return_status status = m_client->read_model_data(m_model_name, m_signature, segments, total_size);
    	if (isGPUPtr(tensors.begin()->get().tensor_data().data())){
        	int iter=0;
        	for(auto &t: tensors){
            		cudaMemcpy((char*)t.get().tensor_data().data(), (char*)segments[iter].first, segments[iter].second, cudaMemcpyHostToDevice);
            		++iter;
        	}
    	}	

        return status.first;
    }



};

}

#endif
