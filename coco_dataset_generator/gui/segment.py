from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import RadioButtons
from matplotlib.path import Path

from PIL import Image
import matplotlib

import argparse
import numpy as np
import glob
import os

import math 

import traceback

from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

from .poly_editor import PolygonInteractor, dist_point_to_segment

import sys
from ..utils.visualize_dataset import return_info

from pprint import pprint

class COCO_dataset_generator(object):

    def __init__(self, fig, ax, args):

        self.ax = ax
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        self.img_dir = args['image_dir']
        self.index = 0
        self.fig = fig
        self.polys = []
        self.zoom_scale, self.points, self.prev, self.submit_p, self.lines, self.circles = 1.2, [], None, None, [], []

        self.zoom_id = fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.click_id = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.clickrel_id = fig.canvas.mpl_connect('button_release_event', self.onclick_release)
        self.keyboard_id = fig.canvas.mpl_connect('key_press_event', self.onkeyboard)

        self.axradio = plt.axes([0.0, 0.0, 0.123, 1])
        self.axbringprev = plt.axes([0.24, 0.05, 0.17, 0.05])
        self.axreset = plt.axes([0.48, 0.05, 0.1, 0.05])
        self.axsubmit = plt.axes([0.59, 0.05, 0.1, 0.05])
        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.05])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.05])
        self.axpan = plt.axes([0.42, 0.05, 0.05, 0.05])
        self.b_bringprev = Button(self.axbringprev, 'Bring Previous Annotations')
        self.b_bringprev.on_clicked(self.bring_prev)
        self.b_reset = Button(self.axreset, 'Reset')
        self.b_reset.on_clicked(self.reset)
        self.b_submit = Button(self.axsubmit, 'Submit')
        self.b_submit.on_clicked(self.submit)
        self.b_next = Button(self.axnext, 'Next')
        self.b_next.on_clicked(self.next)
        self.b_prev = Button(self.axprev, 'Prev')
        self.b_prev.on_clicked(self.previous)
        self.b_pan = Button(self.axpan, 'Pan')
        self.b_pan.on_clicked(self.pan)

        self.button_axes = [self.axbringprev, self.axreset, self.axsubmit, self.axprev, self.axnext, self.axradio, self.axpan]

        self.existing_polys = []
        self.existing_patches = []
        self.selected_poly = False
        self.objects = []
        self.feedback = args['feedback']

        self.right_click = False
        
        self.edit_poly = False

        self.text = ''

        with open(args['class_file'], 'r') as f:
            self.class_names = [x.strip() for x in f.readlines() if x.strip() != ""]

        self.radio = RadioButtons(self.axradio, self.class_names)
        self.class_names = ('BG',) + tuple(self.class_names)

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))

        if len(self.img_paths)==0:
            self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        if os.path.exists(self.img_paths[self.index][:-3]+'txt'):
            self.index = len(glob.glob(os.path.join(self.img_dir, '*.txt')))
        self.checkpoint = self.index
        
        try:
            im = Image.open(self.img_paths[self.index])
        except IndexError:
            print ("Reached end of dataset! Delete some TXT files if you want to relabel some images in the folder")
            exit()

        width, height = im.size
        im.close()

        image = plt.imread(self.img_paths[self.index])
        
        self.pan_flag = True

        if args['feedback']:

            from mask_rcnn import model as modellib
            from mask_rcnn.get_json_config import get_demo_config
            
            #from skimage.measure import find_contours
            from .contours import find_contours

            from mask_rcnn.visualize_cv2 import random_colors
            
            config = get_demo_config(len(self.class_names)-2, True)
            
            if args['config_path'] is not None:
                config.from_json(args['config_path'])

            # Create model object in inference mode.
            model = modellib.MaskRCNN(mode="inference", model_dir='/'.join(args['weights_path'].split('/')[:-2]), config=config)

            # Load weights trained on MS-COCO
            model.load_weights(args['weights_path'], by_name=True)

            r = model.detect([image], verbose=0)[0]

            # Number of instances
            N = r['rois'].shape[0]

            masks = r['masks']

            # Generate random colors
            colors = random_colors(N)

            # Show area outside image boundaries.
            height, width = image.shape[:2]

            class_ids, scores = r['class_ids'], r['scores']

            for i in range(N):
                color = colors[i]

                # Label
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = self.class_names[class_id]

                # Mask
                mask = masks[:, :, i]

                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)
                
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)

                    verts = np.fliplr(verts) - 1
                    pat = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='green', linewidths=0, alpha=0.6)
                    self.ax.add_collection(pat)
                    self.objects.append(label)
                    self.existing_patches.append(pat)
                    self.existing_polys.append(Polygon(verts, closed=True, alpha=0.25, facecolor='red'))

        self.ax.imshow(image, aspect='auto')

        self.text+=str(self.index)+'\n'+os.path.abspath(self.img_paths[self.index])+'\n'+str(width)+' '+str(height)+'\n\n'
        
        self.insert_pt_ptr = None

    def bring_prev(self, event):

        if not self.feedback:

            poly_verts, self.objects = return_info(self.img_paths[self.index-1][:-3]+'txt')

            for num in poly_verts:
                self.existing_polys.append(Polygon(num, closed=True, alpha=0.5, facecolor='red'))

                pat = PatchCollection([Polygon(num, closed=True)], facecolor='green', linewidths=0, alpha=0.6)
                self.ax.add_collection(pat)
                self.existing_patches.append(pat)

    def points_to_polygon(self):
        return np.reshape(np.array(self.points), (int(len(self.points)/2), 2))
    
    def polygon_to_points(self, poly):
        return np.reshape(np.array(poly), -1).tolist()

    def deactivate_all(self):
        self.fig.canvas.mpl_disconnect(self.zoom_id)
        self.fig.canvas.mpl_disconnect(self.click_id)
        self.fig.canvas.mpl_disconnect(self.clickrel_id)
        self.fig.canvas.mpl_disconnect(self.keyboard_id)

    def onkeyboard(self, event):
    
        if not event.inaxes:
            return
        elif event.key == 'a':
            
            if Polygon(
                self.points_to_polygon()).get_path().contains_point(
                (event.xdata, event.ydata)) and self.right_click:
                
                self.p.set_color("purple")
                self.edit_poly = True

            if self.selected_poly:
                
                self.points = self.interactor.get_polygon().xy.flatten()
                self.interactor.deactivate()
                self.right_click = True
                self.selected_poly = False
                self.fig.canvas.mpl_connect(self.click_id, self.onclick)
                self.polygon.color = (0,255,0)
                self.fig.canvas.draw()
            else:
                
                for i, poly in enumerate(self.existing_polys):

                    if poly.get_path().contains_point((event.xdata, event.ydata)):

                        self.radio.set_active(self.class_names.index(self.objects[i])-1)
                        self.polygon = self.existing_polys[i]
                        self.existing_patches[i].set_visible(False)
                        self.fig.canvas.mpl_disconnect(self.click_id)
                        self.ax.add_patch(self.polygon)
                        self.fig.canvas.draw()
                        self.interactor = PolygonInteractor(self.ax, self.polygon)
                        self.selected_poly = True
                        self.existing_polys.pop(i)
                        break

        elif event.key == 'r':

            for i, poly in enumerate(self.existing_polys):
                if poly.get_path().contains_point((event.xdata, event.ydata)):
                    self.existing_patches[i].set_visible(False)
                    self.existing_patches[i].remove()
                    self.existing_patches.pop(i)
                    self.existing_polys.pop(i)
                    break
        
        elif event.key == 'd' and self.edit_poly:
            
            # brute force because python is fast enough anyway!
            
            dist_threshold = 20**2

            poly = self.points_to_polygon()
            poly = poly[:-1] if np.equal(poly[0], poly[-1]).all() else poly
            
            if poly.shape[0] <= 3:
                print ("Cannot have object with lesser sides than triangle!")
                return 
            
            pt_dists = np.sum((poly - np.array([event.xdata, event.ydata]))**2, axis=1)
            remove_idx = np.argmin(pt_dists) 
            remove_idx = -1 if pt_dists[remove_idx] > dist_threshold else remove_idx  
            
            if remove_idx != -1:
                poly = np.delete(poly, remove_idx, 0)

                self.points = self.polygon_to_points(poly)
                self.circles[remove_idx].remove()
                del self.circles[remove_idx]
                
                l = self.lines[remove_idx][0]
                x11, x2 = l._xorig
                y11, y2 = l._yorig
                
                self.lines[remove_idx][0].remove()
                del self.lines[remove_idx]
                
                l = self.lines[remove_idx-1][0]
                x3, x12 = l._xorig
                y3, y12 = l._yorig
                
                assert x11 == x12 and y11 == y12

                self.lines[remove_idx-1][0].remove()
                del self.lines[remove_idx-1]
                
                new_line = self.ax.plot([x3, x2], [y3, y2], 'b--')
                self.lines.insert(remove_idx-1, new_line) 
                
                self.p.set_alpha(0)
                self.p.remove()
                self.p = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='purple', linewidths=0, alpha=0.4)
                self.ax.add_collection(self.p)
                
            self.fig.canvas.draw() 

        elif (event.key == 'i' or event.key == 'o') and self.edit_poly:
            
            # list vs numpy after too much effort :) 
            # detect intersection between lines to create meaningful polygons
            
            # closest pt and y-axis

            self.p.remove()
            self.p = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='purple', linewidths=0, alpha=0.1)
            self.ax.add_collection(self.p)

            poly = self.points_to_polygon()
            lines = [(poly[idx], poly[idx+1]) for idx in range(len(poly)-1)]
            
            dist_fn = lambda pt: abs((pt[0] - event.xdata)**2 + (pt[1] - event.ydata)**2)
            
            # Closest point makes more sense than dist between pt and line segment
            mid_pts = list(map(lambda xy_pts:
               ((xy_pts[0][0] + xy_pts[1][0])*0.5, (xy_pts[0][1] + xy_pts[1][1])*0.5), lines))

            dists = list(map(lambda pt: dist_fn(pt), mid_pts))
            
            if self.insert_pt_ptr == None:
                
                last_pt = poly[-1]
                poly = poly[:-1] # remove last pt as first pt to use advanced robotic API capabilities (human deep learner downsides)
                # code reading redo prompts - understanding the same code over and over again! virtual understanding repetition benefits?
            
                closest_line_idx = np.argmin(dists)
                closest_pt_idx = np.argmin(list(map(lambda pt: dist_fn(pt), 
                    lines[closest_line_idx:closest_line_idx+1])))

                pt_dists = np.sum((poly - np.array([event.xdata, event.ydata]))**2, axis=1)
                v1_idx, v2_idx = np.argpartition(pt_dists, 2)[:2]
                
                #if closest_line_idx + closest_pt_idx + 1 == len(poly):
                #    closest_pt_idx -= 1
                
                self.insert_pt_ptr = {"closest_line_idx": closest_line_idx, "closest_pt_idx": 
                        closest_pt_idx, "poly": poly, "num_added_pts": 0, # "points_of_interest": [],
                        "last_pt": last_pt}
            
            else:
                closest_line_idx = self.insert_pt_ptr["closest_line_idx"]
                closest_pt_idx = self.insert_pt_ptr["closest_pt_idx"]
                self.insert_pt_ptr["last_pt"] = poly[-1]
                poly = poly[:-1]

            # index_pt = closest_pt_idx + 2 if closest_line_idx == len(lines)-1 else closest_pt_idx
            poly = np.vstack((poly[:closest_line_idx + closest_pt_idx + self.insert_pt_ptr["num_added_pts"] + 1, :], 
                np.array([[event.xdata, event.ydata]]), 
                poly[closest_line_idx + closest_pt_idx + self.insert_pt_ptr["num_added_pts"] + 1:])) 
            
            self.insert_pt_ptr["num_added_pts"] += 1
            
            dists = list(map(lambda pt: dist_fn(pt), self.insert_pt_ptr["poly"]))
            
            closest_point_idx = np.argmin(dists)
            # self.insert_pt_ptr["points_of_interest"].append(self.insert_pt_ptr["poly"][closest_point_idx])
            self.insert_pt_ptr["last_pt"] = self.insert_pt_ptr["poly"][closest_point_idx]

            circle = plt.Circle(self.insert_pt_ptr["poly"][closest_point_idx], 2.5, color='red')
            self.ax.add_artist(circle)
            self.circles.append(circle)

            pt_dists = np.sum((poly - np.array([event.xdata, event.ydata]))**2, axis=1)
            v1_idx, v2_idx = np.argpartition(pt_dists, 2)[:2]

            closest_pt = lines[closest_line_idx][closest_pt_idx]
            
            # self.lines[closest_line_idx][0].remove()
            # del self.lines[closest_line_idx]

            #line1 = self.ax.plot([closest_pt[0], event.xdata], [closest_pt[1], event.ydata], 'b--')
            #line2 = self.ax.plot([event.xdata, 
            #                      lines[closest_line_idx][int(closest_pt_idx!=1)][0]], 
            #                     [event.ydata, 
            #                      lines[closest_line_idx][int(closest_pt_idx!=1)][1]], 'b--')
            #self.lines = self.lines[:closest_line_idx] + [line1, line2] + self.lines[closest_line_idx:]
            
            #print (poly)
            poly = poly if np.equal(poly[-1], (self.insert_pt_ptr["last_pt"],)).all() else np.append(poly, (self.insert_pt_ptr["last_pt"],), axis=0)
            #print (poly)

            self.points = self.polygon_to_points(poly)
            
            circle = plt.Circle((event.xdata,event.ydata),2.5,color='black')
            self.ax.add_artist(circle)
            self.circles.append(circle)
   
            self.p.set_alpha(0.1)
            #self.p.remove()
            self.p = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='purple', linewidths=0, alpha=0.1)
            self.ax.add_collection(self.p)

        self.fig.canvas.draw()

    def next(self, event):

        if len(self.text.split('\n'))>5:

            print (self.img_paths[self.index][:-3]+'txt')

            with open(self.img_paths[self.index][:-3]+'txt', "w") as text_file:
                text_file.write(self.text)

        self.ax.clear()

        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        if (self.index<len(self.img_paths)-1):
            self.index += 1
        else:
            exit()

        image = plt.imread(self.img_paths[self.index])
        self.ax.imshow(image, aspect='auto')

        im = Image.open(self.img_paths[self.index])
        width, height = im.size
        im.close()

        self.reset_all()

        self.text+=str(self.index)+'\n'+os.path.abspath(self.img_paths[self.index])+'\n'+str(width)+' '+str(height)+'\n\n'

    def reset_all(self):

        self.polys = []
        self.text = ''
        self.points, self.prev, self.submit_p, self.lines, self.circles = [], None, None, [], []

    def previous(self, event):

        if (self.index>self.checkpoint):
            self.index-=1
        
        # print (self.img_paths[self.index][:-3]+'txt')
        os.remove(self.img_paths[self.index][:-3]+'txt')

        self.ax.clear()

        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        image = plt.imread(self.img_paths[self.index])
        self.ax.imshow(image, aspect='auto')

        im = Image.open(self.img_paths[self.index])
        width, height = im.size
        im.close()

        self.reset_all()

        self.text+=str(self.index)+'\n'+os.path.abspath(self.img_paths[self.index])+'\n'+str(width)+' '+str(height)+'\n\n'

    def onclick(self, event):

        if not event.inaxes:
            return
        if not any([x.in_axes(event) for x in self.button_axes]):
            if event.button==1:
                self.points.extend([event.xdata, event.ydata])
                # print (event.xdata, event.ydata)

                circle = plt.Circle((event.xdata,event.ydata),2.5,color='black')
                self.ax.add_artist(circle)
                self.circles.append(circle)

                if (len(self.points)<4):
                    self.r_x = event.xdata
                    self.r_y = event.ydata
            else:
                if self.edit_poly:
                    self.onkeyboard(None)
                
                if len(self.points)>5:
                    self.right_click=True
                    self.fig.canvas.mpl_disconnect(self.click_id)
                    self.click_id = None
                    self.points.extend([self.points[0], self.points[1]])
                    #self.prev.remove()

            if (len(self.points)>2):
                line = self.ax.plot([self.points[-4], self.points[-2]], [self.points[-3], self.points[-1]], 'b--')
                self.lines.append(line)

            self.fig.canvas.draw()

            if len(self.points)>4:
                if not self.prev is None:
                    try:
                        self.prev.remove()
                    except Exception:
                        pass

                self.p = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='red', linewidths=0, alpha=0.4)
                self.ax.add_collection(self.p)
                self.prev = self.p

                self.fig.canvas.draw()

            #if len(self.points)>4:
            #    print 'AREA OF POLYGON: ', self.find_poly_area(self.points)
            #print event.x, event.y

    def find_poly_area(self):
        coords = self.points_to_polygon()
        x, y = coords[:,0], coords[:,1]
        return (0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))/2 #shoelace algorithm

    def onclick_release(self, event):

        if any([x.in_axes(event) for x in self.button_axes]) or self.selected_poly:
            return

        if hasattr(self, 'r_x') and hasattr(self, 'r_y') and None not in [self.r_x, self.r_y, event.xdata, event.ydata]:
            if np.abs(event.xdata - self.r_x)>10 and np.abs(event.ydata - self.r_y)>10: # 10 pixels limit for rectangle creation
                if len(self.points)<4:

                    self.right_click=True
                    self.fig.canvas.mpl_disconnect(self.click_id)
                    self.click_id = None
                    bbox = [np.min([event.xdata, self.r_x]), np.min([event.ydata, self.r_y]), np.max([event.xdata, self.r_x]), np.max([event.ydata, self.r_y])]
                    self.r_x = self.r_y = None

                    self.points = [bbox[0], bbox[1], bbox[0], bbox[3], bbox[2], bbox[3], bbox[2], bbox[1], bbox[0], bbox[1]]
                    self.p = PatchCollection([Polygon(self.points_to_polygon(), closed=True)], facecolor='red', linewidths=0, alpha=0.4)
                    self.ax.add_collection(self.p)
                    self.fig.canvas.draw()
    
    def pan(self, event):
        
        self.ax.set_xlim(self.cur_xlim)
        self.ax.set_ylim(self.cur_ylim)
        self.ax.figure.canvas.draw()

    def zoom(self, event):

        if not event.inaxes:
            return
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # PAN
        if self.pan_flag:
            self.cur_xlim = cur_xlim
            self.cur_ylim = cur_ylim
            self.pan_flag = False

        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        if event.button == 'down':
            # deal with zoom in
            scale_factor = 1 / self.zoom_scale
        elif event.button == 'up':
            # deal with zoom out
            scale_factor = self.zoom_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print (event.button)

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.ax.figure.canvas.draw()
 
    def reset(self, event):

        if not self.click_id:
            self.click_id = fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        try:
            for line in self.lines:
                line.pop(0).remove()
            for circle in self.circles:
                circle.remove()
            self.lines, self.circles = [], []
            self.points = []
            self.p.remove()
            self.prev = self.p = None
        
        except AttributeError:
            
            #traceback.print_exc()
            
            self.prev = self.p = None
            self.points = []
            
        #print (len(self.lines))
        #print (len(self.circles))

    def print_points(self):

        ret = ''
        for x in self.points:
            ret+='%.2f'%x+' '
        return ret

    def submit(self, event):

        if not self.right_click:
            print ('Right click before submit is a must!!')
        else:

            self.text+=self.radio.value_selected+'\n'+'%.2f'%self.find_poly_area()+'\n'+self.print_points()+'\n\n'
            self.right_click = False
            #print (self.points)

            self.lines, self.circles = [], []
            self.click_id = fig.canvas.mpl_connect('button_press_event', self.onclick)

            self.polys.append(Polygon(self.points_to_polygon(), closed=True, color=np.random.rand(3), alpha=0.4, fill=True))
            if self.submit_p:
                self.submit_p.remove()
            self.submit_p = PatchCollection(self.polys, cmap=matplotlib.cm.jet, alpha=0.4)
            self.ax.add_collection(self.submit_p)
            self.points = []
            
            if self.p:
                self.p.remove()
            
            self.circles = []
            self.edit_poly = False
            self.insert_pt_ptr = None

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Path to the image dir")
    ap.add_argument("-c", "--class_file", required=True, help="Path to the classes file of the dataset")
    ap.add_argument('-w', "--weights_path", default=None, help="Path to Mask RCNN checkpoint save file")
    ap.add_argument('-x', "--config_path", default=None, help="Path to Mask RCNN training config JSON file to load model based on specific parameters")
    args = vars(ap.parse_args())
    
    args['feedback'] = args['weights_path'] is not None

    fig = plt.figure(figsize=(14, 14))
    ax = plt.gca()

    gen = COCO_dataset_generator(fig, ax, args)

    plt.subplots_adjust(bottom=0.2)
    plt.show()

    gen.deactivate_all()
