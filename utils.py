from os import path, listdir, makedirs, walk
from zipfile import ZipFile, ZIP_DEFLATED
from inspect import getfile, currentframe
from smtplib import SMTP
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from base64 import b64decode


def zipFolder(p, pathFunc, zipf):
    for base, dirs, files in walk(p):
        if base.endswith('__pycache__'):
            continue

        for file in files:
            if file.endswith('.tar'):
                continue

            fn = path.join(base, file)
            zipf.write(fn, pathFunc(fn))


def create_exp_dir(resultFolderPath):
    # create folders
    if not path.exists(resultFolderPath):
        makedirs(resultFolderPath)

    codeFilename = 'code.zip'
    zipPath = '{}/{}'.format(resultFolderPath, codeFilename)
    zipf = ZipFile(zipPath, 'w', ZIP_DEFLATED)

    # init project base folder
    baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    # init path function
    pathFunc = lambda fn: path.relpath(fn, baseFolder)
    # init folders we want to zip
    foldersToZip = ['models']
    # save folders files
    for folder in foldersToZip:
        zipFolder('{}/{}'.format(baseFolder, folder), pathFunc, zipf)

    # save main folder files
    foldersToZip = ['.']
    for folder in foldersToZip:
        folderName = '{}/{}'.format(baseFolder, folder)
        for file in listdir(folderName):
            filePath = '{}/{}'.format(folderName, file)
            if path.isfile(filePath):
                zipf.write(filePath, pathFunc(filePath))

    # close zip file
    zipf.close()
    # return zipPath, codeFilename


# msg - email message, MIMEMultipart() object
def attachFiletoEmail(msg, fileFullPath):
    with open(fileFullPath, 'rb') as z:
        # attach file
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(z.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % path.basename(fileFullPath))
        msg.attach(part)


def sendEmail(toAddr, subject, content, attachments=None):
    # init email addresses
    fromAddr = "yochaiz@campus.technion.ac.il"
    # init connection
    server = SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    passwd = b'WXo4Nzk1NzE='
    server.login(fromAddr, b64decode(passwd).decode('utf-8'))
    # init message
    msg = MIMEMultipart()
    msg['From'] = fromAddr
    msg['Subject'] = subject
    msg.attach(MIMEText(content, 'plain'))

    if attachments:
        for att in attachments:
            if path.exists(att):
                if path.isdir(att):
                    for filename in listdir(att):
                        attachFiletoEmail(msg, '{}/{}'.format(att, filename))
                else:
                    attachFiletoEmail(msg, att)

    # send message
    for dst in toAddr:
        msg['To'] = dst
        text = msg.as_string()
        try:
            server.sendmail(fromAddr, dst, text)
        except Exception as e:
            print('Sending email failed, error:[{}]'.format(e))

    server.close()
