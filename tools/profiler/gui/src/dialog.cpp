/*
 * Copyright (C) 2020  GreenWaves Technologies, SAS
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <QtWidgets>
#include <unistd.h>
#include <iostream>

#include "dialog.hpp"

#define MESSAGE \
    Dialog::tr("<p>Message boxes have a caption, a text, " \
               "and any number of buttons, each with standard or custom texts." \
               "<p>Click a button to close the message box. Pressing the Esc button " \
               "will activate the detected escape button (if any).")
#define MESSAGE_DETAILS \
    Dialog::tr("If a message box has detailed text, the user can reveal it " \
               "by pressing the Show Details... button.")


class DialogOptionsWidget : public QGroupBox
{
public:
    explicit DialogOptionsWidget(QWidget *parent = nullptr);

    void addCheckBox(const QString &text, int value);
    void addSpacer();
    int value() const;

private:
    typedef QPair<QCheckBox *, int> CheckBoxEntry;
    QVBoxLayout *layout;
    QList<CheckBoxEntry> checkBoxEntries;
};

DialogOptionsWidget::DialogOptionsWidget(QWidget *parent) :
    QGroupBox(parent) , layout(new QVBoxLayout)
{
    setTitle(Dialog::tr("Options"));
    setLayout(layout);
}

void DialogOptionsWidget::addCheckBox(const QString &text, int value)
{
    QCheckBox *checkBox = new QCheckBox(text);
    layout->addWidget(checkBox);
    checkBoxEntries.append(CheckBoxEntry(checkBox, value));
}

void DialogOptionsWidget::addSpacer()
{
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
}

int DialogOptionsWidget::value() const
{
    int result = 0;
    for (const CheckBoxEntry &checkboxEntry : qAsConst(checkBoxEntries)) {
        if (checkboxEntry.first->isChecked())
            result |= checkboxEntry.second;
    }
    return result;
}

Dialog::Dialog(QWidget *parent)
    : QWidget(parent)
{
    QVBoxLayout *verticalLayout;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        QHBoxLayout *horizontalLayout = new QHBoxLayout(this);
        QGroupBox *groupBox = new QGroupBox(QGuiApplication::applicationDisplayName(), this);
        horizontalLayout->addWidget(groupBox);
        verticalLayout = new QVBoxLayout(groupBox);
    } else {
        verticalLayout = new QVBoxLayout(this);
    }

    QToolBox *toolbox = new QToolBox;
    verticalLayout->addWidget(toolbox);

    int frameStyle = QFrame::Sunken | QFrame::Panel;

    /*integerLabel = new QLabel;
    integerLabel->setFrameStyle(frameStyle);
    QPushButton *integerButton =
            new QPushButton(tr("QInputDialog::get&Int()"));

    doubleLabel = new QLabel;
    doubleLabel->setFrameStyle(frameStyle);
    QPushButton *doubleButton =
            new QPushButton(tr("QInputDialog::get&Double()"));

    itemLabel = new QLabel;
    itemLabel->setFrameStyle(frameStyle);
    QPushButton *itemButton = new QPushButton(tr("QInputDialog::getIte&m()"));

    textLabel = new QLabel;
    textLabel->setFrameStyle(frameStyle);
    QPushButton *textButton = new QPushButton(tr("QInputDialog::get&Text()"));

    multiLineTextLabel = new QLabel;
    multiLineTextLabel->setFrameStyle(frameStyle);
    QPushButton *multiLineTextButton = new QPushButton(tr("QInputDialog::get&MultiLineText()"));

    colorLabel = new QLabel;
    colorLabel->setFrameStyle(frameStyle);
    QPushButton *colorButton = new QPushButton(tr("QColorDialog::get&Color()"));

    fontLabel = new QLabel;
    fontLabel->setFrameStyle(frameStyle);
    QPushButton *fontButton = new QPushButton(tr("QFontDialog::get&Font()"));
*/
    directoryLabel = new QLabel;
    directoryLabel->setFrameStyle(frameStyle);
    QPushButton *directoryButton =
            new QPushButton(tr("QFileDialog::getE&xistingDirectory()"));

    /*openFileNameLabel = new QLabel;
    openFileNameLabel->setFrameStyle(frameStyle);
    QPushButton *openFileNameButton =
            new QPushButton(tr("QFileDialog::get&OpenFileName()"));

    openFileNamesLabel = new QLabel;
    openFileNamesLabel->setFrameStyle(frameStyle);
    QPushButton *openFileNamesButton =
            new QPushButton(tr("QFileDialog::&getOpenFileNames()"));
    */

    execFileNameLabel = new QLabel;
    execFileNameLabel->setFrameStyle(frameStyle);
    QPushButton *execFileNameButton =
            new QPushButton(tr("QFileDialog::get&execFileName()"));

    /*connect(integerButton, &QAbstractButton::clicked, this, &Dialog::setInteger);
    connect(doubleButton, &QAbstractButton::clicked, this, &Dialog::setDouble);
    connect(itemButton, &QAbstractButton::clicked, this, &Dialog::setItem);
    connect(textButton, &QAbstractButton::clicked, this, &Dialog::setText);
    connect(multiLineTextButton, &QAbstractButton::clicked, this, &Dialog::setMultiLineText);
    connect(colorButton, &QAbstractButton::clicked, this, &Dialog::setColor);
    connect(fontButton, &QAbstractButton::clicked, this, &Dialog::setFont);
    */
    connect(directoryButton, &QAbstractButton::clicked,
            this, &Dialog::setExistingDirectory);
    /*connect(openFileNameButton, &QAbstractButton::clicked,
            this, &Dialog::setOpenFileName);
    connect(openFileNamesButton, &QAbstractButton::clicked,
            this, &Dialog::setOpenFileNames);
            */
    connect(execFileNameButton, &QAbstractButton::clicked,
            this, &Dialog::setExecFileName);

    /*layout->setColumnStretch(1, 1);
    layout->setColumnMinimumWidth(1, 250);
    layout->addWidget(integerButton, 0, 0);
    layout->addWidget(integerLabel, 0, 1);
    layout->addWidget(doubleButton, 1, 0);
    layout->addWidget(doubleLabel, 1, 1);
    layout->addWidget(itemButton, 2, 0);
    layout->addWidget(itemLabel, 2, 1);
    layout->addWidget(textButton, 3, 0);
    layout->addWidget(textLabel, 3, 1);
    layout->addWidget(multiLineTextButton, 4, 0);
    layout->addWidget(multiLineTextLabel, 4, 1);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding), 5, 0);
    toolbox->addItem(page, tr("Input Dialogs"));
    */
    const QString doNotUseNativeDialog = tr("Do not use native dialog");
    /*
    page = new QWidget;
    layout = new QGridLayout(page);
    layout->setColumnStretch(1, 1);
    layout->addWidget(colorButton, 0, 0);
    layout->addWidget(colorLabel, 0, 1);
    colorDialogOptionsWidget = new DialogOptionsWidget;
    colorDialogOptionsWidget->addCheckBox(doNotUseNativeDialog, QColorDialog::DontUseNativeDialog);
    colorDialogOptionsWidget->addCheckBox(tr("Show alpha channel") , QColorDialog::ShowAlphaChannel);
    colorDialogOptionsWidget->addCheckBox(tr("No buttons") , QColorDialog::NoButtons);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding), 1, 0);
    layout->addWidget(colorDialogOptionsWidget, 2, 0, 1 ,2);

    toolbox->addItem(page, tr("Color Dialog"));

    page = new QWidget;
    layout = new QGridLayout(page);
    layout->setColumnStretch(1, 1);
    layout->addWidget(fontButton, 0, 0);
    layout->addWidget(fontLabel, 0, 1);
    fontDialogOptionsWidget = new DialogOptionsWidget;
    fontDialogOptionsWidget->addCheckBox(doNotUseNativeDialog, QFontDialog::DontUseNativeDialog);
    fontDialogOptionsWidget->addCheckBox(tr("Show scalable fonts"), QFontDialog::ScalableFonts);
    fontDialogOptionsWidget->addCheckBox(tr("Show non scalable fonts"), QFontDialog::NonScalableFonts);
    fontDialogOptionsWidget->addCheckBox(tr("Show monospaced fonts"), QFontDialog::MonospacedFonts);
    fontDialogOptionsWidget->addCheckBox(tr("Show proportional fonts"), QFontDialog::ProportionalFonts);
    fontDialogOptionsWidget->addCheckBox(tr("No buttons") , QFontDialog::NoButtons);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding), 1, 0);
    layout->addWidget(fontDialogOptionsWidget, 2, 0, 1 ,2);
    toolbox->addItem(page, tr("Font Dialog"));
*/
    QWidget* page = new QWidget;
    QGridLayout* layout = new QGridLayout(page);
    layout->setColumnStretch(1, 1);
    layout->addWidget(directoryButton, 0, 0);
    layout->addWidget(directoryLabel, 0, 1);
    /*layout->addWidget(openFileNameButton, 1, 0);
    layout->addWidget(openFileNameLabel, 1, 1);
    layout->addWidget(openFileNamesButton, 2, 0);
    layout->addWidget(openFileNamesLabel, 2, 1);
    */
    layout->addWidget(execFileNameButton, 1, 0);
    layout->addWidget(execFileNameLabel, 1, 1);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding), 4, 0);
    toolbox->addItem(page, tr("File Dialogs"));

    /*page = new QWidget;
    layout = new QGridLayout(page);
    layout->setColumnStretch(1, 1);
    layout->addWidget(criticalButton, 0, 0);
    layout->addWidget(criticalLabel, 0, 1);
    layout->addWidget(informationButton, 1, 0);
    layout->addWidget(informationLabel, 1, 1);
    layout->addWidget(questionButton, 2, 0);
    layout->addWidget(questionLabel, 2, 1);
    layout->addWidget(warningButton, 3, 0);
    layout->addWidget(warningLabel, 3, 1);
    layout->addWidget(errorButton, 4, 0);
    layout->addWidget(errorLabel, 4, 1);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding), 5, 0);
    toolbox->addItem(page, tr("Message Boxes"));
    */
    //setWindowTitle(QGuiApplication::applicationDisplayName());
}

/*void Dialog::setInteger()
{
//! [0]
    bool ok;
    int i = QInputDialog::getInt(this, tr("QInputDialog::getInt()"),
                                 tr("Percentage:"), 25, 0, 100, 1, &ok);
    if (ok)
        integerLabel->setText(tr("%1%").arg(i));
//! [0]
}

void Dialog::setDouble()
{
//! [1]
    bool ok;
    double d = QInputDialog::getDouble(this, tr("QInputDialog::getDouble()"),
                                       tr("Amount:"), 37.56, -10000, 10000, 2, &ok);
    if (ok)
        doubleLabel->setText(QString("$%1").arg(d));
//! [1]
}

void Dialog::setItem()
{
//! [2]
    QStringList items;
    items << tr("Spring") << tr("Summer") << tr("Fall") << tr("Winter");
QFileDialog daliog(this);
  dialog.setFileMode(QFileDialog::Directory);
  dialog.setViewMode(QFileDialog::Detail);
  QStringList fileNames;
  if (dialog.exec())
    fileNames = dialog.selectedFiles();("User name:"), QLineEdit::Normal,
                                         QDir::home().dirName(), &ok);
    if (ok && !text.isEmpty())
        textLabel->setText(text);
//! [3]
}

void Dialog::setMultiLineText()
{
//! [4]
    bool ok;
    QString text = QInputDialog::getMultiLineText(this, tr("QInputDialog::getMultiLineText()"),
                                                  tr("Address:"), "John Doe\nFreedom Street", &ok);
    if (ok && !text.isEmpty())
        multiLineTextLabel->setText(text);
//! [4]
}

void Dialog::setColor()
{
    const QColorDialog::ColorDialogOptions options = QFlag(colorDialogOptionsWidget->value());
    const QColor color = QColorDialog::getColor(Qt::green, this, "Select Color", options);

    if (color.isValid()) {
        colorLabel->setText(color.name());
        colorLabel->setPalette(QPalette(color));
        colorLabel->setAutoFillBackground(true);
    }
}

void Dialog::setFont()
{
    const QFontDialog::FontDialogOptions options = QFlag(fontDialogOptionsWidget->value());
    bool ok;
    QFont font = QFontDialog::getFont(&ok, QFont(fontLabel->text()), this, "Select Font", options);
    if (ok) {
        fontLabel->setText(font.key());
        fontLabel->setFont(font);
    }
}
*/
void Dialog::setExistingDirectory()
{
    std::cout << "**** setExistingDirectory *****" << std::endl;
    QFileDialog::Options options = QFlag(fileDialogOptionsWidget->value());
    options |= QFileDialog::DontResolveSymlinks | QFileDialog::ShowDirsOnly;
    std::cout << "**** setExistingDirectory 1 *****" << std::endl;
    QString directory = QFileDialog::getExistingDirectory(this,
                                tr("QFileDialog::getExistingDirectory()"),
                                directoryLabel->text(),
                                options);
    std::cout << "**** setExistingDirectory 2 *****" << std::endl;
    if (!directory.isEmpty())
        directoryLabel->setText(directory);
    std::cout << "**** setExistingDirectory 3 *****" << std::endl;
}
/*
void Dialog::setOpenFileName()
{
    const QFileDialog::Options options = QFlag(fileDialogOptionsWidget->value());
    QString selectedFilter;
    QString fileName = QFileDialog::getOpenFileName(this,
                                tr("QFileDialog::getOpenFileName()"),
                                openFileNameLabel->text(),
                                tr("All Files (*);;Text Files (*.txt)"),
                                &selectedFilter,
                                options);
    if (!fileName.isEmpty())
        openFileNameLabel->setText(fileName);
}

void Dialog::setOpenFileNames()
{
    const QFileDialog::Options options = QFlag(fileDialogOptionsWidget->value());
    QString selectedFilter;
    QStringList files = QFileDialog::getOpenFileNames(
                                this, tr("QFileDialog::getOpenFileNames()"),
                                openFilesPath,
                                tr("All Files (*);;Text Files (*.txt)"),
                                &selectedFilter,
                                options);
    if (files.count()) {
        openFilesPath = files[0];
        openFileNamesLabel->setText(QString("[%1]").arg(files.join(", ")));
    }
}
*/

void Dialog::setExecFileName()
{
    const QFileDialog::Options options = QFlag(fileDialogOptionsWidget->value());
    QString selectedFilter;
    QString fileName = QFileDialog::getOpenFileName(this,
                                tr("QFileDialog::getExecFileName()"),
                                execFileNameLabel->text(),
                                tr("All Files (*)"),
                                &selectedFilter,
                                options);
    if (!fileName.isEmpty())
        execFileNameLabel->setText(fileName);
}
/*
void Dialog::criticalMessage()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::critical(this, tr("QMessageBox::critical()"),
                                    MESSAGE,
                                    QMessageBox::Abort | QMessageBox::Retry | QMessageBox::Ignore);
    if (reply == QMessageBox::Abort)
        criticalLabel->setText(tr("Abort"));
    else if (reply == QMessageBox::Retry)
        criticalLabel->setText(tr("Retry"));
    else
        criticalLabel->setText(tr("Ignore"));
}

void Dialog::informationMessage()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::information(this, tr("QMessageBox::information()"), MESSAGE);
    if (reply == QMessageBox::Ok)
        informationLabel->setText(tr("OK"));
    else
        informationLabel->setText(tr("Escape"));
}

void Dialog::questionMessage()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, tr("QMessageBox::question()"),
                                    MESSAGE,
                                    QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
    if (reply == QMessageBox::Yes)
        questionLabel->setText(tr("Yes"));
    else if (reply == QMessageBox::No)
        questionLabel->setText(tr("No"));
    else
        questionLabel->setText(tr("Cancel"));
}

void Dialog::warningMessage()
{
    QMessageBox msgBox(QMessageBox::Warning, tr("QMessageBox::warning()"),
                       MESSAGE, nullptr, this);
    msgBox.setDetailedText(MESSAGE_DETAILS);
    msgBox.addButton(tr("Save &Again"), QMessageBox::AcceptRole);
    msgBox.addButton(tr("&Continue"), QMessageBox::RejectRole);
    if (msgBox.exec() == QMessageBox::AcceptRole)
        warningLabel->setText(tr("Save Again"));
    else
        warningLabel->setText(tr("Continue"));

}

void Dialog::errorMessage()
{
    errorMessageDialog->showMessage(
            tr("This dialog shows and remembers error messages. "
               "If the checkbox is checked (as it is by default), "
               "the shown message will be shown again, "
               "but if the user unchecks the box the message "
               "will not appear again if QErrorMessage::showMessage() "
               "is called with the same message."));
    errorLabel->setText(tr("If the box is unchecked, the message "
                           "won't appear again."));
}
*/
