#ifndef __FLAMESTORE_SERVER_CONTEXT_H
#define __FLAMESTORE_SERVER_CONTEXT_H

#include <thallium.hpp>

namespace flamestore {

namespace tl = thallium;

struct ServerContext {
    tl::engine*     m_engine = nullptr;
};

}

#endif
