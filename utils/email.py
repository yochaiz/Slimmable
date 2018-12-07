from os import path, listdir
from smtplib import SMTP
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from base64 import b64decode


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
