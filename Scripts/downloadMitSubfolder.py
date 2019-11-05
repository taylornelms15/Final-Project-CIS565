import sys
import os
import re
import argparse
import wget


import pdb

mitIndexFileName = "mitFileIndex.txt"

def parseIndexFile():
    lines = []
    with open(mitIndexFileName, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]#take off ending newline, any other leading/following text
    return lines

def downloadMitSet(sourceURL, outputRoot, indexLines):
    print("Downloading recursively from [%s] into folder [%s]" % (sourceURL, outputRoot))
    if not os.path.exists(outputRoot):
        os.makedirs(outputRoot)

    matchingLines = [x for x in indexLines if x.startswith(sourceURL)]
    prefix = "http://"

    for line in matchingLines:
        fullURL = prefix + line

        pathFromRoot = line[len(sourceURL):]
        splitPathFromRoot = re.split('/', pathFromRoot)
        if len(splitPathFromRoot) > 1:
            folderFromRoot = "/".join(splitPathFromRoot[:-1])
            os.makedirs(os.path.join(outputRoot, folderFromRoot))
        outputPath = os.path.join(outputRoot, pathFromRoot)
        print("\tDownloading from [%s] into [%s]" % (fullURL, outputPath))
        wget.download(fullURL, outputPath)

def main():
    parser      = argparse.ArgumentParser()
    parser.add_argument("src", help = "Source directory or URL from which to download")
    parser.add_argument("-o", "--output", help = "Optional name for output directory")
    args        = parser.parse_args()
    src         = args.src
    output      = args.output

    if src[:7] == "http://":
        src = src[7:]
    elif src[:8] == "https://":
        src = src[8:]
    if not output:
        srcSplit = re.split('/', src)
        if len(srcSplit[-1]):
            output = srcSplit[-1]
        else:
            output = srcSplit[-2]

    indexLines = parseIndexFile()

    downloadMitSet(src, output, indexLines)


if __name__ == "__main__":
    main()
