from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import *
from kivy.core.window import Window

from numpy.linalg import inv
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import subprocess
import cv2
import numpy as np
import math

from threading import Thread



class CamApp(App):

	didBackgroundDepthSaved = False
	didBackgroundDepthLoaded = False
	arrayOfBackgroundDepth = []

	def build(self):
		#using kinect
		self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
		#opencv2 stuffs
		#self.capture = cv2.VideoCapture(0)
		#self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
		#self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		# cv2.namedWindow("CV2 Image")

		self.img1=Image()
		layout = FloatLayout()
		layout.add_widget(self.img1)	

		#image layer for game
		# self.img2=Image()
		# layout.add_widget(self.img2)
		# self.rectPos = [Window.size[0]/2, Window.size[1]/2]

		#scale window to max
		# Window.fullscreen = True
		Window.show_cursor = False
		
		Clock.schedule_interval(self.update, 1.0/25.0)
		return layout

	def update(self, dt):
		frame = None
		if self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame():
			
			depthFrame = self._kinect.get_last_depth_frame()
			# depthFrame = self.draw_depth_SandMount_frame(depthFrame)
			if self.didBackgroundDepthSaved == True:
				depthFrame = self.draw_depth_SandMount_frame(depthFrame)
				depthFrame = np.flip(depthFrame,1)
				depthFrame = cv2.flip(depthFrame, 0)
				texture = Texture.create(size=(self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), colorfmt='bgra')
				texture.blit_buffer(depthFrame.tostring(), colorfmt='bgra', bufferfmt='ubyte')
				self.img1.texture = texture
				
			if self.didBackgroundDepthSaved == False:
				self.save_depth_frame(depthFrame)
				self.didBackgroundDepthSaved = True

			depthFrame = None

		# for bgra frame
		# 	frame = self._kinect.get_last_color_frame()
		# 	frame = np.asanyarray(frame, dtype=np.uint8)
		# 	frame.shape = (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width,4)
		# if frame is not None:

		# 	frame = np.flip(frame,1)
			
		# 	frame = cv2.flip(frame, 0)
		# 	texture = Texture.create(size=(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), colorfmt='bgra')
		# 	texture.blit_buffer(frame.tostring(), colorfmt='bgra', bufferfmt='ubyte')
			
		# 	self.img1.texture = texture

	def save_depth_frame(self, frame):
		f = open('Background', 'w+b')
		print(frame)
		binary_format = bytearray(frame)
		f.write(binary_format)
		f.close()

	def load_depth_frame(self):
		if self.didBackgroundDepthLoaded == False:
			f=open('Background',"rb")
			self.arrayOfBackgroundDepth = np.frombuffer(f.read(), dtype=np.uint16)
			self.arrayOfBackgroundDepth = self.opencvProcessing(self.arrayOfBackgroundDepth)
			self.arrayOfBackgroundDepth = np.reshape(self.arrayOfBackgroundDepth, (self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, ))
			print("arrayOfBackgroundDepth")
			print(self.arrayOfBackgroundDepth)
			f.close()
			self.didBackgroundDepthLoaded = True

	def opencvProcessing(self, objectHeights):
		objectHeights = cv2.morphologyEx(objectHeights, cv2.MORPH_OPEN, np.ones((5,5), np.uint16))
		return objectHeights

	def draw_depth_SandMount_frame(self, frame):
		self.load_depth_frame()
		# get an array of heights (np.uint16) of any objects on top of the background, and flipping its left and right sides
		objectHeights = self.arrayOfBackgroundDepth - frame + 90
		objectHeights = np.reshape(objectHeights, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
		# if height between 0 50 then from green to yellow, if height between 50 100 then from yellow to red
		# f8Red = np.uint8(frame.clip(1,4000)/16.)
		# f8Green = np.uint8(frame.clip(1,4000)/16.)
		# f8Blue = np.uint8(frame.clip(1,4000)/16.)
		f8Red = np.uint8((objectHeights * 50/18).clip(0,250))
		f8Green = np.uint8((objectHeights * -3 + 500).clip(0,250))
		f8Blue = np.uint8((objectHeights.clip(0,1)))
		f8Redint = np.reshape(f8Red, (self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, ))
		f8Greenint = np.reshape(f8Green, (self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, ))
		f8Blueint = np.reshape(f8Blue, (self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, ))
		f8Alphaint = np.full((self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width), 255, dtype=np.uint8)
		frame8bit = np.dstack((f8Blueint, f8Greenint, f8Redint, f8Alphaint))
		frame8bit.shape = (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height, 4)

		return frame8bit

if __name__ == '__main__':
	CamApp().run()
	#self._kinect.close()