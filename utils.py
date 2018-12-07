from os import path, listdir, makedirs, walk
from zipfile import ZipFile, ZIP_DEFLATED
from inspect import getfile, currentframe


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
