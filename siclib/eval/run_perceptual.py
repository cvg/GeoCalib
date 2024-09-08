"""Run the TPAMI 2023 Perceptual model.

Run the model of the paper
    A Deep Perceptual Measure for Lens and Camera Calibration, TPAMI 2023
    https://lvsn.github.io/deepcalib/
through the public Dashboard available at http://rachmaninoff.gel.ulaval.ca:8005.
"""

import argparse
import json
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm

# mypy: ignore-errors

JS_DROP_FILES = "var k=arguments,d=k[0],g=k[1],c=k[2],m=d.ownerDocument||document;for(var e=0;;){var f=d.getBoundingClientRect(),b=f.left+(g||(f.width/2)),a=f.top+(c||(f.height/2)),h=m.elementFromPoint(b,a);if(h&&d.contains(h)){break}if(++e>1){var j=new Error('Element not interactable');j.code=15;throw j}d.scrollIntoView({behavior:'instant',block:'center',inline:'center'})}var l=m.createElement('INPUT');l.setAttribute('type','file');l.setAttribute('multiple','');l.setAttribute('style','position:fixed;z-index:2147483647;left:0;top:0;');l.onchange=function(q){l.parentElement.removeChild(l);q.stopPropagation();var r={constructor:DataTransfer,effectAllowed:'all',dropEffect:'none',types:['Files'],files:l.files,setData:function u(){},getData:function o(){},clearData:function s(){},setDragImage:function i(){}};if(window.DataTransferItemList){r.items=Object.setPrototypeOf(Array.prototype.map.call(l.files,function(x){return{constructor:DataTransferItem,kind:'file',type:x.type,getAsFile:function v(){return x},getAsString:function y(A){var z=new FileReader();z.onload=function(B){A(B.target.result)};z.readAsText(x)},webkitGetAsEntry:function w(){return{constructor:FileSystemFileEntry,name:x.name,fullPath:'/'+x.name,isFile:true,isDirectory:false,file:function z(A){A(x)}}}}}),{constructor:DataTransferItemList,add:function t(){},clear:function p(){},remove:function n(){}})}['dragenter','dragover','drop'].forEach(function(v){var w=m.createEvent('DragEvent');w.initMouseEvent(v,true,true,m.defaultView,0,0,0,b,a,false,false,false,false,0,null);Object.setPrototypeOf(w,null);w.dataTransfer=r;Object.setPrototypeOf(w,DragEvent.prototype);h.dispatchEvent(w)})};m.documentElement.appendChild(l);l.getBoundingClientRect();return l"  # noqa E501


def setup_driver():
    """Setup the Selenium browser."""
    options = webdriver.FirefoxOptions()
    geckodriver_path = "/snap/bin/geckodriver"  # specify the path to your geckodriver
    driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)
    return webdriver.Firefox(options=options, service=driver_service)


def run(args):
    """Run on an image folder."""
    driver = setup_driver()
    driver.get("http://rachmaninoff.gel.ulaval.ca:8005/")
    time.sleep(5)
    result_div = driver.find_element(By.ID, "estimated-parameters-display")

    def upload_image(path):
        path = Path(path).absolute().as_posix()
        elem = driver.find_element(By.ID, "dash-uploader")
        inp = driver.execute_script(JS_DROP_FILES, elem, 25, 25)
        inp._execute("sendKeysToElement", {"value": [path], "text": path})

    def run_image(path, prev_result, timeout_seconds=60):
        # One main assumption is that subsequent images will have different results
        # from each other, otherwise we cannot detect that the inference has completed.
        upload_image(path)
        started = time.time()
        while True:
            result = result_div.text
            if (result != prev_result) and result:
                return result
            prev_result = result
            if (time.time() - started) > timeout_seconds:
                raise TimeoutError

    result = str(result_div.text)
    number = r"(nan|-?\d*\.?\d*)"
    pattern = re.compile(
        f"Pitch: {number}° / Roll: {number}° / HFOV : {number}° / Distortion: {number}"
    )

    paths = sorted(args.images.iterdir())
    results = {}
    for path in (pbar := tqdm(paths)):
        pbar.set_description(path.name)
        result = run_image(path, result)
        match = pattern.match(result)
        if match is None:
            print("Error, cannot parse:", result, path)
            continue
        results[path.name] = tuple(map(float, match.groups()))

    args.results.write_text(json.dumps(results))
    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    run(args)
