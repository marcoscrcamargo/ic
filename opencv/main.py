import knn
import mlp
import svm

def main():

# Parametros:
	# img_path - caminho para a imagem
	# imshow = true/false(default) para mostrar janela com imagem
	# info = true/false(default) para mostrar informações


	mlp_ret = mlp.classify('inputs/0/0.jpg', imshow=True)
	knn_ret = knn.classify('inputs/0/0.jpg')
	svm_ret = svm.classify('inputs/0/0.jpg')

	#Retorno:
		# O retorno da função é um dicionário contendo dois valores: 'pxl' e 'hst',
		# que indica os resultados de cada método.
		# 'pxl' e 'hst' também é um dicionário, contendo os seguintes valores:
		# {'label':label da imagem, '0':prob de ser 0 ,'1':prob de ser 1, '2': prob de ser 2 }

	print("KNN\n")
	print("Label: " + str(knn_ret['pxl']['label']) +
			" prob:" + str(knn_ret['pxl'][str(knn_ret['pxl']['label'])]))
	print("Label: " + str(knn_ret['hst']['label']) +
			" prob:" + str(knn_ret['hst'][str(knn_ret['hst']['label'])]))	
	print("SVM\n")
	print("Label: " + str(svm_ret['pxl']['label']) +
			" prob:" + str(svm_ret['pxl'][str(svm_ret['pxl']['label'])]))
	print("Label: " + str(svm_ret['hst']['label']) +
			" prob:" + str(svm_ret['hst'][str(svm_ret['hst']['label'])]))	
	print("MLP\n")
	print("Label: " + str(mlp_ret['pxl']['label']) +
			" prob:" + str(mlp_ret['pxl'][str(mlp_ret['pxl']['label'])]))
	print("Label: " + str(mlp_ret['hst']['label']) +
			" prob:" + str(mlp_ret['hst'][str(mlp_ret['hst']['label'])]))	



if __name__ == "__main__":
	main()