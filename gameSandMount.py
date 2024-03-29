from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import *
from kivy.core.window import Window

# for getting screen resolution only
from tkinter import Tk

from numpy.linalg import inv
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import subprocess
import cv2
import numpy as np
import math

from random import seed
from random import randint

from ar_markers import detect_markers

# from threading import Thread



class SandMountApp(App):

	didBackgroundDepthSaved = False
	didBackgroundDepthLoaded = False
	didBackgroundCropped = False
	didProjectorKinectMatched = False
	hasProjectorKinectFirstPointFound = False
	depthOffsetX = 0
	depthOffsetY = 0
	depthLimitedWidth = 512
	depthLimitedHeight = 414
	arrayOfBackgroundDepth = []
	screenWidth = 1000
	screenHeight = 1000
	tempIntX = 0
	tempIntY = 0
	arrayOfWaterDrop = []

	def on_stop(self):
		print('on_stop triggered')
		self.layout.remove_widget(self.img1)

	def build(self):
		#using kinect
		self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

		# getting screen resolution
		root = Tk()
		self.screenWidth = root.winfo_screenwidth()
		self.screenHeight = root.winfo_screenheight()

		# AR marker
		self.exitMarker = [1]
		self.calibrateMarkerCode = [3, 4, 9, 3727]
		self.calibrateMarkerDepthPosition = {}
		self.calibrateMarkerBGRAPosition = {}
		self.projectorKinectMatchingMarkerCode = [991, 4090]
		self.projectorKinectMatchingMarkerDepthPosition = {}
		self.projectorKinectMatchingMarkerScreenPosition = {}
		self.projectorKinectMatchingMarkerBGRAPosition = {}

		self.img1=Image()
		self.layout = BoxLayout()
		self.layout.add_widget(self.img1)	

		#scale window to max
		# Window.fullscreen = True
		Window.show_cursor = True
		
		Clock.schedule_interval(self.update, 1.0/15.0)
		return self.layout

	def update(self, dt):

		# After two calibration markers are detected, cropping the sand container out from screen
		if self.didBackgroundCropped == False:
			if 3 in self.calibrateMarkerDepthPosition and 9 in self.calibrateMarkerDepthPosition:
				print(self.calibrateMarkerDepthPosition)
				self.depthOffsetX = self.calibrateMarkerDepthPosition[3][0] + 72
				self.depthOffsetY = self.calibrateMarkerDepthPosition[3][1] + 45
				bottomRightPointX = self.calibrateMarkerDepthPosition[9][0]
				bottomRightPointY = self.calibrateMarkerDepthPosition[9][1]
				self.depthLimitedWidth = bottomRightPointX - self.depthOffsetX + 40
				self.depthLimitedHeight = bottomRightPointY - self.depthOffsetY -50
				self.didBackgroundCropped = True
				# Window.left = 522
				# Window.top = 204
				# Window.size = (904, 599)
				self.arrayOfWaterDrop = np.zeros((self.depthLimitedHeight, self.depthLimitedWidth), dtype=np.uint8)

		#After two projector kinect calibration markers are detected, we set the X Y offset and size of window 
		if (991 in self.projectorKinectMatchingMarkerDepthPosition and 4090 in self.projectorKinectMatchingMarkerDepthPosition)  and self.didProjectorKinectMatched == False:
			print(self.projectorKinectMatchingMarkerScreenPosition)
			print(self.projectorKinectMatchingMarkerDepthPosition)
			diffWidthDepthPtBtwTwoCalibrationPt  = self.projectorKinectMatchingMarkerDepthPosition[4090][0] - self.projectorKinectMatchingMarkerDepthPosition[991][0]
			diffHeightDepthPtBtwTwoCalibrationPt = self.projectorKinectMatchingMarkerDepthPosition[4090][1] - self.projectorKinectMatchingMarkerDepthPosition[991][1]
			diffWidthDepthPtBtwLastCalibrationPtAndTopleftCornerOfSandMount =  self.depthOffsetX - self.projectorKinectMatchingMarkerDepthPosition[4090][0]
			diffHeightDepthPtBtwLastCalibrationPtAndTopleftCornerOfSandMount =  self.depthOffsetY - self.projectorKinectMatchingMarkerDepthPosition[4090][1]
			diffWidthScreenPtBtwTwoCalibrationPt = self.projectorKinectMatchingMarkerScreenPosition[4090][0] -  self.projectorKinectMatchingMarkerScreenPosition[991][0]
			diffHeightScreenPtBtwTwoCalibrationPt = self.projectorKinectMatchingMarkerScreenPosition[4090][1] -  self.projectorKinectMatchingMarkerScreenPosition[991][1]
			diffWidthScreenPtBtwLastCalibrationPtAndTopleftCornerOfSandMount = diffWidthScreenPtBtwTwoCalibrationPt/diffWidthDepthPtBtwTwoCalibrationPt*diffWidthDepthPtBtwLastCalibrationPtAndTopleftCornerOfSandMount
			diffHeightScreenPtBtwLastCalibrationPtAndTopleftCornerOfSandMount = diffHeightScreenPtBtwTwoCalibrationPt/diffHeightDepthPtBtwTwoCalibrationPt*diffHeightDepthPtBtwLastCalibrationPtAndTopleftCornerOfSandMount
			screenPointXTopLeftCornerSandMount = diffWidthScreenPtBtwLastCalibrationPtAndTopleftCornerOfSandMount + diffWidthScreenPtBtwTwoCalibrationPt
			screenPointYTopLeftCornerSandMount = diffHeightScreenPtBtwLastCalibrationPtAndTopleftCornerOfSandMount + diffHeightScreenPtBtwTwoCalibrationPt
			diffWidthDepthPtBtwTopLeftCornerAndBottomRightCornerOfSandMount = self.depthLimitedWidth
			diffHeightDepthPtBtwTopLeftCornerAndBottomRightCornerOfSandMount = self.depthLimitedHeight
			diffWidthScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount =   diffWidthScreenPtBtwTwoCalibrationPt / diffWidthDepthPtBtwTwoCalibrationPt * diffWidthDepthPtBtwTopLeftCornerAndBottomRightCornerOfSandMount
			diffHeightScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount = diffHeightScreenPtBtwTwoCalibrationPt / diffHeightDepthPtBtwTwoCalibrationPt * diffHeightDepthPtBtwTopLeftCornerAndBottomRightCornerOfSandMount
			# screenPointXBottomRightCornerSandMount = screenPointXTopLeftCornerSandMount +  diffWidthScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount
			# screenPointYBottomRightCornerSandMount = screenPointYTopLeftCornerSandMount + diffHeightScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount

			Window.left = screenPointXTopLeftCornerSandMount -39
			Window.top = screenPointYTopLeftCornerSandMount + 95
			Window.size = (diffWidthScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount, diffHeightScreenPtBtwTopLeftCornerAndBottomRightCornerOfSandMount)
			# Window.left = 522
			# Window.top = 204
			# Window.size = (904, 599)
			print(Window.left, Window.top)
			print(Window.size)
			self.didProjectorKinectMatched = True
			


		frame = None
		if self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame():
			
		# for depth frame
			depthFrame = self._kinect.get_last_depth_frame()
		# for rgb frame
			frame = self._kinect.get_last_color_frame()
			frame = np.asanyarray(frame, dtype=np.uint8)
			frame.shape = (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width,4)
			# depthFrame = self.draw_depth_SandMount_frame(depthFrame)
			if self.didBackgroundDepthSaved == True and self.didBackgroundCropped == True and self.didProjectorKinectMatched == True:
				depthFrame = np.reshape(depthFrame, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
				depthFrame = np.flip(depthFrame,1) #leftToRight
				# depthFrame = cv2.flip(depthFrame, 0) #UpsideDown
				depthFrameProcessed = self.draw_depth_SandMount_frame(depthFrame)
				# I don't know why imShow do not need the following cvFlip to display the image correctly, but to use kivy (openGL) texture
				# we'd better flip it upside down with cv2.flip here
				depthFrameProcessed = cv2.flip(depthFrameProcessed, 0) #UpsideDown
				# The below 3 lines are for displaying
				texture = Texture.create(size=(self.depthLimitedWidth, self.depthLimitedHeight), colorfmt='bgra')
				texture.blit_buffer(depthFrameProcessed.tostring(), colorfmt='bgra', bufferfmt='ubyte')
				self.img1.texture = texture
				# self explanatory
				self.img1.allow_stretch = True
				if frame is not None:
					frame = np.flip(frame, 1)
					self.find_markers(frame, depthFrame)
				# print(Window.left)
				# print(Window.top)
				# print(Window.size)
				
			if self.didBackgroundDepthSaved == False and self.didBackgroundCropped == True:
				depthFrame = np.reshape(depthFrame, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
				depthFrame = np.flip(depthFrame,1)
				# depthFrame = cv2.flip(depthFrame, 0)
				self.save_depth_frame(depthFrame)
				self.didBackgroundDepthSaved = True

			if self.didBackgroundCropped == False:
				#if both frame and depthFrame are available, we will then find the two ar markers for calibration
				if frame is not None:
					frame = np.flip(frame,1)
					if depthFrame is not None:
						depthFrame = np.reshape(depthFrame, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
						depthFrame = np.flip(depthFrame,1)
						# depthFrame = cv2.flip(depthFrame, 0) #UpsideDown
						self.find_markers(frame, depthFrame)
					# I don't know why imShow do not need the following cvFlip to display the image correctly, but to use kivy (openGL) texture
					# we'd better flip it upside down with cv2.flip here
					frame = cv2.flip(frame, 0)
				# The below 3 lines are for displaying
				# texture = Texture.create(size=(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), colorfmt='bgra')
				# texture.blit_buffer(frame.tostring(), colorfmt='bgra', bufferfmt='ubyte')
				# self.img1.texture = texture

			if self.didProjectorKinectMatched == False and self.didBackgroundCropped == True:
				# print('Window Position and Size:')
				# print(Window.left)
				# print(Window.top)
				# print(Window.size)
				# print(self.screenWidth, self.screenHeight)
				if self.hasProjectorKinectFirstPointFound == False:
					if self.tempIntY < self.screenHeight:
						self.tempIntY = self.tempIntY + 1
					else:
						self.tempIntY = 0
					Window.top = self.tempIntY
					if self.tempIntY == 0:
						self.tempIntX = self.tempIntX + 1
					if self.tempIntX >= self.screenWidth:
						self.tempIntX = 0
					Window.left = self.tempIntX
					Window.size = (200, 200)
					self.img1.source = 'marker_991.png'
				else:
					if self.tempIntY < self.screenHeight:
						self.tempIntY = self.tempIntY + 1
					else:
						self.tempIntY = 0
					Window.top = self.tempIntY
					if self.tempIntY == 0:
						self.tempIntX = self.tempIntX + 1
					if self.tempIntX >= self.screenWidth:
						self.tempIntX = 0
					Window.left = self.tempIntX
					Window.size = (200, 200)
					self.img1.source = 'marker_3314.png'
				if frame is not None and depthFrame is not None:
					frame = np.flip(frame,1)
					grey =  cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
					ret, dst= cv2.threshold(grey,250,255,cv2.THRESH_BINARY)
					# cv2.imshow("Image", dst)
					self.find_markers(dst, depthFrame)


			depthFrame = None


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
		# get an array of heights (np.uint16) of any objects on top of the background
		objectHeights = self.arrayOfBackgroundDepth - frame + 90
		# Subsetting the depthFrame by applying offsets and limited width and limited height
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
		frame8bit.shape = (self.depthLimitedHeight, self.depthLimitedWidth, 4)
		frame8bit = self.addWaterDrop(objectHeights, frame8bit)
		self.moveWaterDrop(objectHeights)
		return frame8bit

	def find_markers(self, frame, depthframe):
		# Detect marker(s) from colorFrame
		markers = detect_markers(frame)

		# Using comMethod to call foreign library through pyKinectRunTime -> pyKinectV2
		# Obtain C Pointer array of 1080*1920 for mapping of DepthPoints to ColorPoints
		ptr_depth = np.ctypeslib.as_ctypes(depthframe.flatten())
		L = depthframe.size
		ColorPointsSize = 1080*1920
		TYPE_DepthSpacePointArray = PyKinectV2._DepthSpacePoint * ColorPointsSize
		csps1 = TYPE_DepthSpacePointArray()
		error_state = self._kinect._mapper.MapColorFrameToDepthSpace(L, ptr_depth, ColorPointsSize, csps1)
		if error_state is not 0:
			print(error_state)

		# Change array of pointers to np.array and reshape it
		mappingDepthtoColor = np.ctypeslib.as_array(csps1)
		mappingDepthtoColor.shape = (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width)

		# Save markers' coordination in depthFrame
		for marker in markers:
		#	marker.highlite_marker(frame)
		#	self.label.text = str(marker.id) + str(marker.center)
			if (marker.id not in self.calibrateMarkerCode):
				print(marker.id)

			if marker.id in self.calibrateMarkerCode and (not 3 in self.calibrateMarkerDepthPosition or not 9 in self.calibrateMarkerDepthPosition):
				# self.calibrateMarkerCode.append(marker.id)
				print(marker.id, marker.center, sep=": ")
				depthPoint = mappingDepthtoColor[marker.center[1]][marker.center[0]]
				print(depthPoint)
				try:
					depthPointInt = (int(depthPoint[0]), int(depthPoint[1]))
					self.calibrateMarkerDepthPosition[marker.id] = depthPointInt
				except:
					print('cannot locate depthPoint of ' + str(marker.id))

			if marker.id in self.projectorKinectMatchingMarkerCode:
				print(marker.id, marker.center, sep=": ")
				depthPoint = mappingDepthtoColor[marker.center[1]][marker.center[0]]
				print(depthPoint)
				try:
					if (991 not in self.projectorKinectMatchingMarkerDepthPosition and marker.id == 991):
						screenPoint = (self.tempIntX, self.tempIntY)
						self.projectorKinectMatchingMarkerScreenPosition[marker.id] =  screenPoint
						self.tempIntX = self.tempIntX + 1200
						self.tempIntY = self.tempIntY + 200
						depthPointInt = (int(depthPoint[0]), int(depthPoint[1]))
						self.projectorKinectMatchingMarkerDepthPosition[marker.id] = depthPointInt
					if (4090 not in self.projectorKinectMatchingMarkerDepthPosition and marker.id == 4090):
						screenPoint = (self.tempIntX, self.tempIntY)
						self.projectorKinectMatchingMarkerScreenPosition[marker.id] =  screenPoint
						depthPointInt = (int(depthPoint[0]), int(depthPoint[1]))
						self.projectorKinectMatchingMarkerDepthPosition[marker.id] = depthPointInt
					self.hasProjectorKinectFirstPointFound = True
				except:
					print('cannot locate depthPoint of ' + str(marker.id))

			if marker.id in self.exitMarker:
				print('Exit marker found, exiting the game...')
				print(self)
				# self.stop()



			

	def addWaterDrop(self, objectHeights, frame8bit, offsetX=0, offsetY=0, width=0, height=0, thickness=0):
		objectHeightsint = objectHeights.astype(int)
		# for objHeight, cellOfWaterDrop in np.nditer([objectHeightsint, self.arrayOfWaterDrop], op_flags=['readwrite']):
		#     if objHeight > 350 and objHeight < 400:
		#         cellOfWaterDrop[...] +=np.uint8(20)
		self.arrayOfWaterDrop[np.logical_and((objectHeightsint) > 350, (objectHeightsint)< 400)] += np.uint8(1)
		arrayOfWaterDrop = self.arrayOfWaterDrop.clip(0, 10)
		WaterDepthColorBlue = np.uint8((arrayOfWaterDrop*20).clip(0,255))
		WaterDepthColor = np.zeros((self.depthLimitedHeight, self.depthLimitedWidth), dtype=np.uint8)
		f8Alphaint = np.uint8((arrayOfWaterDrop*20).clip(0,255))
		f8AlphaintFull = np.full((self.depthLimitedHeight, self.depthLimitedWidth), 255, dtype=np.uint8)
		f8Alphaintx4 = np.dstack((f8Alphaint, f8Alphaint, f8Alphaint, f8Alphaint))/255
		Waterframe8bit = np.dstack((WaterDepthColorBlue, WaterDepthColor, WaterDepthColor, f8AlphaintFull))
		# cv2.imshow('WaterFrame',Waterframe8bit)
		WaterCVmat = frame8bit * (1 - f8Alphaintx4) + Waterframe8bit * f8Alphaintx4
		WaterCVmat = WaterCVmat.astype(np.uint8)
		return WaterCVmat

	def moveWaterDrop(self, objectHeights):
		objectHeightsint = objectHeights.astype(int)
		for y in range(1, objectHeights.shape[0]-1):
			for x in range(1, objectHeights.shape[1]-1):
				objHeightsAtWaterDrop = objectHeightsint[y-1:y+2, x-1:x+2]
				for k in range(0,self.arrayOfWaterDrop[y, x]):
					waterDepth = self.arrayOfWaterDrop[y-1:y+2, x-1:x+2]
					totalHeightWithWaterDepth = objHeightsAtWaterDrop + waterDepth
					minHeight = np.amin(totalHeightWithWaterDepth)
					for i in range(0, 9):
						n = int(i/3)
						m = int(i%3)
						if totalHeightWithWaterDepth[n, m] == minHeight:
							self.arrayOfWaterDrop[y, x] = self.arrayOfWaterDrop[y, x] - 1
							self.arrayOfWaterDrop[y+n-1, x+m-1] = self.arrayOfWaterDrop[y+n-1, x+m-1] + 1
							break


				# objHeightsAtWaterDrop = objectHeightsint[y-1:y+2, x-1:x+2]
				# waterDepth = self.arrayOfWaterDrop[y-1:y+2, x-1:x+2]
				# sumWaterDepth = waterDepth.sum()
				# if sumWaterDepth > 0:
				# 	whichMatrixRemoved = [1,1,1,1,1,1,1,1,1]
				# 	averageHeight = 0
				# 	# isMatrixRemoved = False
				# 	notWaterDropDistributed = True
				# 	while notWaterDropDistributed:
				# 		notMatrixRemoved = True
				# 		sumWhichMatrixRemoved = sum(whichMatrixRemoved)
				# 		sumObjHeightAtWaterDrop = objHeightsAtWaterDrop.sum()
				# 		averageHeight = (sumObjHeightAtWaterDrop + sumWaterDepth) / (sumWhichMatrixRemoved if sumWhichMatrixRemoved != 0 else 1)
				# 		for i in range(0, 9):
				# 			n = int(i/3)
				# 			m = int(i%3)
				# 			if averageHeight < objHeightsAtWaterDrop[n, m]:
				# 				# objHeightsAtWaterDrop[n, m] = 0
				# 				notMatrixRemoved = False
				# 				whichMatrixRemoved[i] = 0
				# 		if notMatrixRemoved:
				# 			sumWaterDropDistributed = 0
				# 			for i in range(0, 9):
				# 				n = int(i/3)
				# 				m = int(i%3)
				# 				if whichMatrixRemoved[i] == 1:
				# 					self.arrayOfWaterDrop[y+n-1, x+m-1] = np.uint8(averageHeight - objHeightsAtWaterDrop[n, m])
				# 					sumWaterDropDistributed += averageHeight - objHeightsAtWaterDrop[n, m]
				# 				else:
				# 					self.arrayOfWaterDrop[y+n-1, x+m-1] = np.uint8(0)
				# 			remainingWater = sumWaterDepth - sumWaterDropDistributed
				# 			seed(1)
				# 			random = randint(0,8)
				# 			self.arrayOfWaterDrop[y+int(random/3)-1, x+int(random%3)-1] += np.uint8(remainingWater if remainingWater >=0 else 0)
				# 			notWaterDropDistributed = False

if __name__ == '__main__':
	SandMountApp().run()
	#self._kinect.close()