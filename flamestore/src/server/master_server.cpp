#include "master_server.hpp"
#include <fstream>

namespace flamestore {

namespace py11 = pybind11;
namespace tl = thallium;

MasterServer::MasterServer(pymargo_instance_id mid,
           const std::string& workspace_path,
           const std::string& backend_name,
           const std::string& logfile, int loglevel,
           const backend_config_t& backend_config)
: m_engine(CAPSULE2MID(mid))
, m_workspace_path(workspace_path) {
    // Setting up logging
   /* if(logfile.size() != 0) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile, true);
        //m_logger = std::make_unique<spdlog::logger>("FlameStore", file_sink);
    } else {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        //m_logger = std::make_unique<spdlog::logger>("FlameStore", console_sink);
    }*/
    //m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] %v");
    //m_logger->set_level(static_cast<spdlog::level::level_enum>(loglevel));
    //m_logger->info("Initializing MasterProvider at address {}", (std::string)(m_engine.self()));
    //m_logger->info("Workspace is {}", m_workspace_path);
    // Creating the SSG group
    _init_ssg();
    // Setting up the finalize callbacks
    m_engine.push_prefinalize_callback([this]() {
            //m_logger->trace("Pre-finalizing...");
//            m_provider->backend()->on_shutdown();
            _ssg_finalize();
            });
    m_engine.push_finalize_callback([this]() {
            //m_logger->trace("Finalizing...");
            m_provider.reset();
            //m_logger->trace("MasterProvider destroyed");
            });
    // Setting up the MasterProvider
    m_engine.enable_remote_shutdown();
    m_provider = std::make_unique<MasterProvider>(m_engine); //m_logger.get());
    // Setting up the backend
    m_server_context.m_engine = &m_engine;
    //m_server_context.//m_logger = m_logger.get();
    //m_logger->info("Setting up backend as \"{}\"", backend_name);
    m_provider->backend() = AbstractServerBackend::create(
            backend_name, m_server_context, backend_config);//, //m_logger.get());
}

std::string MasterServer::get_connection_info() const {
    return m_engine.self();
}

MasterServer::~MasterServer() {
    //m_logger->debug("Destroying server instance");
}

void MasterServer::_init_ssg() {
    // Initialize ssg
    //m_logger->debug("Initializing SSG");
    int ret = ssg_init();
    if(ret != 0) {
        //m_logger->critical("Could not initialize SSG (ssg_init returned error code {})", ret);
        throw std::runtime_error("Could not initialize SSG");
    }
    // Creating SSG group
    //m_logger->debug("Creating SSG group");
    ssg_group_config_t g_conf = SSG_GROUP_CONFIG_INITIALIZER;
    g_conf.swim_period_length_ms = 1000; /* 1-second period length */
    g_conf.swim_suspect_timeout_periods = 4; /* 4-period suspicion length */
    g_conf.swim_subgroup_member_count = 3; /* 3-member subgroups for SWIM */
    std::string my_address = m_engine.self();
    std::vector<const char*> group_addr_strs = { my_address.c_str() };
    ret = ssg_group_create(m_engine.get_margo_instance(),
            "flamestore", group_addr_strs.data(), 1, &g_conf,
            _ssg_membership_update, (void*)this, &m_ssg_gid);
    if(m_ssg_gid == SSG_GROUP_ID_INVALID) {
        //m_logger->critical("ssg_group_create failed");
        throw std::runtime_error("Could not create SSG group, ssg_group_create failed");
    }
    // Writing the group info to a file
    std::string filename = m_workspace_path + "/.flamestore/group.ssg";
    //m_logger->debug("Storing SSG group info into file {}",filename);
    ret = ssg_group_id_store(filename.c_str(), m_ssg_gid, 1);
    if(ret != 0) {
        //m_logger->critical("Could not store SSG group in workspace (ssg_group_id_store returned {})", ret);
        throw std::runtime_error("Could not store SSG group");
    }
    // Writing the id of the master into a file
    filename =  m_workspace_path + "/.flamestore/master.ssg.id";
    //m_logger->debug("Storing SSG master id info into file {}",filename);
    {
        ssg_member_id_t id;
	auto ret = ssg_get_self_id(m_engine.get_margo_instance(),  &id);
        std::ofstream f(filename);
        f << id;
    }
}

void MasterServer::_ssg_finalize() {
    //m_logger->debug("Destroying SSG group");
    int ret = ssg_group_destroy(m_ssg_gid);
    if(ret != 0) {
        //m_logger->error("SSG could not destroy group (ssg_group_destroy returned error code {})", ret);
    }
    //m_logger->debug("Finalizing SSG");
    ret = ssg_finalize();
    if(ret != 0) {
        //m_logger->error("SSG could not be finalized (ssg_finalize returned error code {})", ret);
    }
    //m_logger->debug("SSG finalized");
}

void MasterServer::_ssg_membership_update(void* arg,
        ssg_member_id_t member_id,
        ssg_member_update_type_t update_type) {
    MasterServer* server = static_cast<MasterServer*>(arg);
    if(update_type == SSG_MEMBER_JOINED) {
        //server->//m_logger->info("Member {} joined", member_id);
        hg_addr_t addr;
        auto ret = ssg_get_group_member_addr(server->m_ssg_gid, member_id, &addr);
        server->m_provider->backend()->on_worker_joined(static_cast<uint64_t>(member_id), addr);
    } else if(update_type == SSG_MEMBER_LEFT) {
        //server->//m_logger->info("Member {} left", member_id);
        server->m_provider->backend()->on_worker_left(member_id);
    } else {
        //server->//m_logger->info("Member {} died", member_id);
        server->m_provider->backend()->on_worker_died(member_id);
    }
}

}
