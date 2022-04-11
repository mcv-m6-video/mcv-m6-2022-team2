import json

def saveData(path, data):
    """
    Saves data to a json file
    :param path:
    :param data:
    :return:
    """
    with open(str(path), 'w') as metadataFile:
        json.dump(data, metadataFile)


def loadData(path):
    """
    Loads data from a json file
    :param path:
    :return:
    """
    with open(str(path), "r") as metadataFile:
        data = json.load(metadataFile)
    return data