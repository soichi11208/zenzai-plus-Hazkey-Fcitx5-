#ifndef HAZKEY_SETTINGS_CONTROLLERS_WARNING_WIDGET_FACTORY_H_
#define HAZKEY_SETTINGS_CONTROLLERS_WARNING_WIDGET_FACTORY_H_

#include <QWidget>
#include <functional>

namespace hazkey::settings {

class WarningWidgetFactory {
   public:
    static QWidget* create(const QString& message,
                           const QString& backgroundColor,
                           const QString& buttonText = QString(),
                           std::function<void()> buttonCallback = nullptr);
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_WARNING_WIDGET_FACTORY_H_
