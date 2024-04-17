
import os

import sys
import time
import weight_block
from tqdm import trange

import random
import torch

from torch.optim import Adam
from torch.autograd import Variable
from grad_loss import Gradloss
import utils

from net import NestFuse_light2_nodense
from net_MRSCFusion import Fusion_network
from args_change import args
import pytorch_msssim




EPSILON = 1e-5


def main():
	original_imgs_path, _ = utils.list_images(args.dataset_ir)
	train_num = 30000
	original_imgs_path = original_imgs_path[:train_num]
	random.seed(1)
	random.shuffle(original_imgs_path)
	# True - RGB , False - gray
	img_flag =False
	train(original_imgs_path, img_flag)



def train(original_imgs_path, img_flag):

	batch_size = args.batch_size
	# load network model
	nc = 1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208]
	f_type = 'res'

	with torch.no_grad():
		deepsupervision = False
		nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
		model_path = args.resume_nestfuse
		print(model_path)
		# load auto-encoder network
		print('Resuming, initializing auto-encoder using weight from {}.'.format(model_path))
		nest_model.load_state_dict(torch.load(model_path))
		nest_model.cuda()
		print(next(nest_model.parameters()).device)
		nest_model.eval()

	# fusion network
	fusion_model = Fusion_network(nb_filter, f_type)
	print(fusion_model)
	fusion_model.cuda()
	print(next(fusion_model.parameters()).device)
	fusion_model.train()

	if args.resume_fusion_model is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
		fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
	optimizer = Adam(fusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	L1_loss = torch.nn.L1Loss()
	ssim_loss = pytorch_msssim.msssim

	tbar = trange(args.epochs)
	print('Start training.....')
	mode = args.mode
	print(mode)
	# creating save path
	temp_path_model = os.path.join(args.save_fusion_model)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:  # models/train/fusionnet/
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:  # models/train/loss_fusionnet/
		os.mkdir(temp_path_loss)

	temp_path_model_w = os.path.join(args.save_fusion_model, mode)  # models/train/fusionnet/
	temp_path_loss_w  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model_w) is False:
		os.makedirs(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	all_loss_grad_value = 0.
	grad_loss =Gradloss()
	for e in tbar:
		print('Epoch %d.....' % e)
		if(e == 1):
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr*0.5
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size,)
		count = 0
		nest_model.cuda()
		fusion_model.cuda()

		# save model
		save_model_filename = mode + ".model"
		save_model_path = os.path.join(temp_path_model_w, save_model_filename)
		file = open('log.txt', 'a')

		for batch in range(batches):
			print("第{}个batch".format(batch))
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_PET = utils.get_train_images(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)
			image_paths_vis = [x.replace('CT', 'MRI') for x in image_paths_ir]
			img_MRI = utils.get_train_images(image_paths_vis, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			count += 1
			optimizer.zero_grad()

			img_PET = Variable(img_PET, requires_grad=False)
			img_MRI = Variable(img_MRI, requires_grad=False)

			img_PET = img_PET.cuda()
			img_MRI = img_MRI.cuda()



			# encoder
			en_PET = nest_model.encoder(img_PET)
			en_MRI = nest_model.encoder(img_MRI)

			# fusion
			f = fusion_model(en_PET, en_MRI)

			# decoder
			outputs = nest_model.decoder_eval(f)


			x_PET = Variable(img_PET.data.clone(), requires_grad=False)########################################
			x_MRI = Variable(img_MRI.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################

			wb = weight_block.weight_block(en_PET[0], en_MRI[0], 3000)
			loss1_value = 0.
			loss2_value = 0.
			loss_grad_value = 0.

			for output in outputs:
				output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
				output = output *255


				g2_ir_fea = en_PET
				g2_vi_fea = en_MRI
				g2_fuse_fea = f
				# ---------------------- LOSS IMAGES ------------------------------------

				ssim_loss_temp1 = ssim_loss(output, x_PET, normalize=True)
				ssim_loss_temp2 = ssim_loss(output, x_MRI, normalize=True)
				loss1_value +=1000*wb[0]* (1 -ssim_loss_temp1 )+1000*wb[1]* (1 -ssim_loss_temp2 )


				w_fea = [1, 10, 100, 1000]
				for ii in range(4):
					g2_ir_temp = g2_ir_fea[ii]
					g2_vi_temp = g2_vi_fea[ii]
					g2_fuse_temp = g2_fuse_fea[ii]
					loss2_value += w_fea[ii]*L1_loss(g2_fuse_temp, 10 * wb[0] * g2_ir_temp + 10 * wb[1] * g2_vi_temp)


				loss_grad_value =grad_loss(img_MRI,img_PET,output)  #


			loss1_value /= len(outputs)
			loss2_value /= len(outputs)
			total_loss = loss1_value +loss2_value +loss_grad_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item()
			all_ssim_loss += loss1_value.item()
			all_loss_grad_value += loss_grad_value.item()

			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t ms-ssim loss: {:.6f}\t fea loss: {:.6f}\t grad loss: {:.6f}\t total: {:.6f},\twb0: {:.6f},\twb1: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  all_loss_grad_value/ args.log_interval,
								  ( all_ssim_loss + all_fea_loss + all_loss_grad_value) / args.log_interval,
								wb[0],
								wb[1]
				)
				tbar.set_description(mesg)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.
				all_loss_grad_value = 0.

				torch.save(fusion_model.state_dict(), save_model_path)
				msg = mesg + '\n'
				file.write(msg)
	#	scheduler.step()
		torch.save(fusion_model.state_dict(), save_model_path)

		file.write("\nDone, trained model saved at{}\n\n\n".format(save_model_path))
		print("\nDone, trained model saved at", save_model_path)
		file.close()

def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":

	main()

