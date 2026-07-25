import os
import numpy as np

from xml.etree import ElementTree as ET

def read_tags(path):
    root = ET.parse(path).getroot()
    tags = [obj.find('name').text for obj in root.findall('object')]
    return list(np.unique(tags))

def write_tags(path, tags, dummy_box: list):
    write_xml(path, [dummy_box for _ in tags], tags, [{} for _ in tags])

def write_xml(xml_path, bboxes=[], tags=[], attributes=[]):
    if os.path.isfile(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
    else:
        root = ET.Element("annotation")
        tree = ET.ElementTree(root)
    
    # for object
    for bbox, tag, attr in zip(bboxes, tags, attributes):
        object = ET.Element("object")

        ET.SubElement(object, "name").text = str(tag)
        """ example
        ET.SubElement(object, "truncated").text = '0'
        ET.SubElement(object, "occluded").text = '0'
        ET.SubElement(object, "difficult").text = '0'
        """
        for key in attr.keys():
            ET.SubElement(object, key).text = attr[key]

        bndbox = ET.Element("bndbox")

        ET.SubElement(bndbox, "xmin").text = '{:.2f}'.format(bbox[0])
        ET.SubElement(bndbox, "ymin").text = '{:.2f}'.format(bbox[1])
        ET.SubElement(bndbox, "xmax").text = '{:.2f}'.format(bbox[2])
        ET.SubElement(bndbox, "ymax").text = '{:.2f}'.format(bbox[3])

        object.append(bndbox)

        root.append(object)

    indent(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)

def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    tags = []
    # attributes = []
    
    for obj in root.findall('object'):
        tag = obj.find('name').text
        bbox = obj.find('bndbox')

        bbox_xmin = float(bbox.find('xmin').text)
        bbox_ymin = float(bbox.find('ymin').text)
        bbox_xmax = float(bbox.find('xmax').text)
        bbox_ymax = float(bbox.find('ymax').text)
        
        # attribute = obj.find('attributes').findall('attribute')[0].find('value').text
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        tags.append(tag)
        # attributes.append(attribute)
    
    return bboxes, tags # , attributes

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i