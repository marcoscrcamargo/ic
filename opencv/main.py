import knn
import mlp
import svm

def main():

# Parametros:
	# img_path - caminho para a imagem
	# imshow = true/false(default) para mostrar janela com imagem
	# inf = true/false(default) para mostrar informações


	print("MLP\n")
	mlp.classify('inputs/0/0.jpg', imshow=True)
	print("KNN\n")
	knn.classify('inputs/0/0.jpg')
	print("SVM\n")
	svm.classify('inputs/0/0.jpg')

if __name__ == "__main__":
	main()