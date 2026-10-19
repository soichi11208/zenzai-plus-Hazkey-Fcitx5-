#ifndef HAZKEY_SETTINGS_CONTROLLERS_TAB_CONTEXT_H_
#define HAZKEY_SETTINGS_CONTROLLERS_TAB_CONTEXT_H_

#include "config.pb.h"
#include "serverconnector.h"

namespace hazkey::settings {

struct TabContext {
    hazkey::config::CurrentConfig* currentConfig{nullptr};
    hazkey::config::Profile* currentProfile{nullptr};
    ServerConnector* server{nullptr};
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_TAB_CONTEXT_H_
