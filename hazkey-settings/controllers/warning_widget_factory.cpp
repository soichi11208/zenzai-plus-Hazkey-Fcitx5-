#include "controllers/warning_widget_factory.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>

namespace hazkey::settings {

QWidget* WarningWidgetFactory::create(const QString& message,
                                      const QString& backgroundColor,
                                      const QString& buttonText,
                                      std::function<void()> buttonCallback) {
    QWidget* warningWidget = new QWidget();
    warningWidget->setStyleSheet(
        QString("background-color: %1; padding: 5px;").arg(backgroundColor));
    QHBoxLayout* warningLayout = new QHBoxLayout(warningWidget);

    QLabel* warningLabel = new QLabel(message);
    warningLabel->setWordWrap(true);
    warningLabel->setStyleSheet("color: black;");
    warningLayout->addWidget(warningLabel);

    if (!buttonText.isEmpty() && buttonCallback) {
        QPushButton* button = new QPushButton(buttonText);
        QObject::connect(button, &QPushButton::clicked, warningWidget,
                         [buttonCallback]() { buttonCallback(); });
        warningLayout->addWidget(button);
    }

    return warningWidget;
}

}  // namespace hazkey::settings
