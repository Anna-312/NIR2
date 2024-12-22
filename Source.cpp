#define _CRT_SECURE_NO_WARNINGS
#include "pch.h"
#include <mysql.h>
#include <cmath>
#include<torch/torch.h>
#include<torch/script.h>


extern "C" __declspec(dllexport) int run_model(UDF_INIT* initid, UDF_ARGS* args, char* is_null, char* error)
{
	torch::NoGradGuard no_grad;
	//Загрузка модели из файла
	torch::jit::script::Module module = torch::jit::load("simple_nn.pth");
	//Преобразование входных данных в тензор
	float mas[] = { *((int*)args->args[0])  };
	torch::Tensor tensor = torch::from_blob(mas, { 1 }, torch::kFloat32);
	//Получение редсказания
	at::Tensor output = module.forward({ tensor }).toTensor();
	float result = output.item<float>();
	int int_result = round(result);
	return int_result;
}

extern "C" __declspec(dllexport) bool run_model_init(UDF_INIT* initid, UDF_ARGS* args, char* message) {
	if (args->arg_count != 1)
	{
		strcpy(message, "run_model requires one argument");
		return 1;
	}
	if (args->arg_type[0] != INT_RESULT)
	{
		strcpy(message, "run_model requires an integer");
		return 1;

	}
	return 0;
}

extern "C" __declspec(dllexport) void run_model_deinit(UDF_INIT* initid) {
	
}
