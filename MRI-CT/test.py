# test phase
import os
from random import random

import torch
from torch.autograd import Variable

from net import NestFuse_light2_nodense
import utils
from args_change import args
import numpy as np
from net_MRSCFusion import Fusion_network
def load_model(path_auto, flag_img=False):
	if flag_img is True:
		nc = 1
	else:
		nc =1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208]

	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
	nest_model.load_state_dict(torch.load(path_auto))
	fs_type = 'res'

	fusion_model =Fusion_network(nb_filter, fs_type)
	print(fusion_model)
	net = torch.load("models/train/fusionnet/fusion_axial/fusion_axial.model")
	fusion_model.load_state_dict(net)
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	fusion_model.eval()
	nest_model.cuda()
	fusion_model.cuda()
	return nest_model, fusion_model
import time
def run_demo(nest_model, fusion_model, ir_path, vis_path, output_path_root, name_ir):
	img_ir, h, w, c = utils.get_test_image(ir_path,flag=False)  # True for rgb
	img_vis, h, w, c = utils.get_test_image(vis_path,flag=False)

	if c is 1:
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vis = img_vis.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vis = Variable(img_vis, requires_grad=False)
		print("img_ir.shape:"+str(img_ir.shape))
		sum = 0
		for i in range(10):
				# encoder
			t1 = time.time()
			en_ir = nest_model.encoder(img_ir)
			en_vis = nest_model.encoder(img_vis)

			# fusion
			f = fusion_model(en_ir, en_vis)
			# decoder
			img_fusion_list = nest_model.decoder_eval(f)
			t1 = time.time() - t1
			sum = sum+t1
			print('t1', t1)
		print(sum/10.)
	else:
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vis_temp = img_vis[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vis_temp = img_vis_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)
			img_vis_temp = Variable(img_vis_temp, requires_grad=False)


			# encoder
			en_ir = nest_model.encoder(img_ir)
			en_vis = nest_model.encoder(img_vis)
			# fusion
			f = fusion_model(en_ir, en_vis)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	# ########################### multi-outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = 'fused_' + '_' + name_ir
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)# save images
		print("output_path:"+output_path)

def main():
	# ################# gray scale ########################################CT_other
	test_path = "images/CT_other/"
	path_auto = args.resume_nestfuse
	output_path_root = "./outputs/CT_MRI/"
	if os.path.exists(output_path_root) is False:
		os.mkdir(output_path_root)


	with torch.no_grad():


		for i in range(1):
			temp = str(i)
			output_path1 = output_path_root + temp + '/'

			if os.path.exists(output_path1) is False:
				os.mkdir(output_path1)
			if os.path.exists(output_path1) is False:
				os.mkdir(output_path1)
			output_path = output_path1
			model, fusion_model = load_model(path_auto)
			imgs_paths_ir, names = utils.Mylist_images(test_path)
			num = len(imgs_paths_ir)

			for i in range(num):
				name_ir = names[i]
				ir_path = imgs_paths_ir[i]
				vis_path = ir_path.replace('CT_other\\', 'MRI_other\\')
				if vis_path.__contains__('CT_other'):
					vis_path = vis_path.replace('CT_other', 'MRI_other')
				else:
					vis_path = vis_path.replace('c.', 'm.')
				run_demo(model, fusion_model, ir_path, vis_path, output_path, name_ir)
			print('Done......')

if __name__ == '__main__':
	main()
