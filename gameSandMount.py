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

from ar_markers import detect_markers

from threading import Thread



class SandMountApp(App):

	didBackgroundDepthSaved = False
	didBackgroundDepthLoaded = False
	didBackgroundCropped = False
	depthOffsetX = 0
	depthOffsetY = 0
	depthLimitedWidth = 512
	depthLimitedHeight = 414
	arrayOfBackgroundDepth = []

	def build(self):
		#using kinect
		self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

		# AR marker
		self.calibrateMarkerCode = [1, 3,4,9, 3727]
		self.calibrateMarkerDepthPosition = {}

		self.img1=Image()
		layout = FloatLayout()
		layout.add_widget(self.img1)	

		#scale window to max
		# Window.fullscreen = True
		Window.show_cursor = False
		
		Clock.schedule_interval(self.update, 1.0/25.0)
		return layout

	def update(self, dt):
		frame = None
		if self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame():
			
		# for depth frame
			depthFrame = self._kinect.get_last_depth_frame()
			# depthFrame = self.draw_depth_SandMount_frame(depthFrame)
			if self.didBackgroundDepthSaved == True and self.didBackgroundCropped == True:
				depthFrame = np.reshape(depthFrame, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
				depthFrame = np.flip(depthFrame,1) #leftToRight
				depthFrame = cv2.flip(depthFrame, 0) #UpsideDown
				depthFrameProcessed = self.draw_depth_SandMount_frame(depthFrame)
				# depthFrameProcessed = np.flip(depthFrameProcessed,1)
				# depthFrameProcessed = cv2.flip(depthFrameProcessed, 0)
				# The below 3 lines are for displaying
				texture = Texture.create(size=(self.depthLimitedWidth, self.depthLimitedHeight), colorfmt='bgra')
				texture.blit_buffer(depthFrameProcessed.tostring(), colorfmt='bgra', bufferfmt='ubyte')
				self.img1.texture = texture
				
			if self.didBackgroundDepthSaved == False:
				depthFrame = np.reshape(depthFrame, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
				depthFrame = np.flip(depthFrame,1)
				depthFrame = cv2.flip(depthFrame, 0)
				self.save_depth_frame(depthFrame)
				self.didBackgroundDepthSaved = True

			if self.didBackgroundCropped == False:
			# for bgra frame
				frame = self._kinect.get_last_color_frame()
				frame = np.asanyarray(frame, dtype=np.uint8)
				frame.shape = (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width,4)
				if frame is not None:

					frame = np.flip(frame,1)
					if depthFrame is not None:
						# depthFrame = np.flip(depthFrame,1)
						self.find_markers(frame, depthFrame)
					frame = cv2.flip(frame, 0)
				# The below 3 lines are for displaying
				# texture = Texture.create(size=(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), colorfmt='bgra')
				# texture.blit_buffer(frame.tostring(), colorfmt='bgra', bufferfmt='ubyte')
				# self.img1.texture = texture

			depthFrame = None

		if self.didBackgroundCropped == False:
			if len(self.calibrateMarkerDepthPosition) >= 2:
				print(self.calibrateMarkerDepthPosition)
				self.depthOffsetX = self.calibrateMarkerDepthPosition[1][0] + 70
				self.depthOffsetY = self._kinect.depth_frame_desc.Height - self.calibrateMarkerDepthPosition[9][1] + 45
				bottomRightPointX = self.calibrateMarkerDepthPosition[9][0]
				bottomRightPointY = self._kinect.depth_frame_desc.Height - self.calibrateMarkerDepthPosition[1][1]
				self.depthLimitedWidth = bottomRightPointX - self.depthOffsetX + 50
				self.depthLimitedHeight = bottomRightPointY - self.depthOffsetY - 50
				self.didBackgroundCropped = True


	def save_depth_frame(self, frame):
		f = open('Background', 'w+b')
		print("Saving arrayOfBackgroundDepth")
		print(frame)
		binary_format = bytearray(frame)
		f.write(binary_format)
		f.close()

	def load_depth_frame(self):
		if self.didBackgroundDepthLoaded == False:
			f=open('Background',"rb")
			self.arrayOfBackgroundDepth = np.frombuffer(f.read(), dtype=np.uint16)
			self.arrayOfBackgroundDepth = self.opencvProcessing(self.arrayOfBackgroundDepth)
			self.arrayOfBackgroundDepth = np.reshape(self.arrayOfBackgroundDepth, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
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
		# objectHeights = np.reshape(objectHeights, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
		objectHeights = objectHeights[self.depthOffsetY:(self.depthOffsetY + self.depthLimitedHeight), self.depthOffsetX:(self.depthOffsetX + self.depthLimitedWidth)]
		# if height between 0 50 then from green to yellow, if height between 50 100 then from yellow to red
		# f8Red = np.uint8(frame.clip(1,4000)/16.)
		# f8Green = np.uint8(frame.clip(1,4000)/16.)
		# f8Blue = np.uint8(frame.clip(1,4000)/16.)
		f8Red = np.uint8((objectHeights * 50/18).clip(0,250))
		f8Green = np.uint8((objectHeights * -3 + 500).clip(0,250))
		f8Blue = np.uint8((objectHeights.clip(0,1)))
		f8Redint = np.reshape(f8Red, (self.depthLimitedHeight * self.depthLimitedWidth, ))
		f8Greenint = np.reshape(f8Green, (self.depthLimitedHeight * self.depthLimitedWidth, ))
		f8Blueint = np.reshape(f8Blue, (self.depthLimitedHeight * self.depthLimitedWidth, ))
		f8Alphaint = np.full((self.depthLimitedHeight * self.depthLimitedWidth), 255, dtype=np.uint8)
		frame8bit = np.dstack((f8Blueint, f8Greenint, f8Redint, f8Alphaint))
		frame8bit.shape = (self.depthLimitedWidth, self.depthLimitedHeight, 4)

		return frame8bit

	def find_markers(self, frame, depthframe):
		# self.calibrateMarkerCode = list()
		markers = detect_markers(frame)

		ptr_depth = np.ctypeslib.as_ctypes(depthframe.flatten())
		L = depthframe.size
		ColorPointsSize = 1080*1920
		TYPE_DepthSpacePointArray = PyKinectV2._DepthSpacePoint * ColorPointsSize
		csps1 = TYPE_DepthSpacePointArray()
		error_state = self._kinect._mapper.MapColorFrameToDepthSpace(L, ptr_depth, ColorPointsSize, csps1)
		if error_state is not 0:
			print(error_state)

		mappingDepthtoColor = np.ctypeslib.as_array(csps1)
		mappingDepthtoColor.shape = (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width)

		for marker in markers:
		#	marker.highlite_marker(frame)
		#	self.label.text = str(marker.id) + str(marker.center)
			if marker.id in self.calibrateMarkerCode:
				# self.calibrateMarkerCode.append(marker.id)
				print(marker.id, marker.center, sep=": ")
				depthPoint = mappingDepthtoColor[marker.center[1]][marker.center[0]]
				print(depthPoint)
				# self.calibrateMarkerDepthPosition[marker.id] = depthPoint
				# isinstance(depthPoint[0], np.float32) and 
				if not ((depthPoint[0] is np.float32('NaN')) | (depthPoint[0] is np.float32('Inf')) | (depthPoint[0] is np.float32('-Inf'))):
					depthPointInt = (int(depthPoint[0]), int(depthPoint[1]))
					self.calibrateMarkerDepthPosition[marker.id] = depthPointInt

if __name__ == '__main__':
	SandMountApp().run()
	#self._kinect.close()