import sys
import time
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

class Craftmlt25kTextDet:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = '10.9.3.241:2814', # default gRPC port
                triton_model_name = 'craft_text_det',
                verbose = False,
                ssl = False,
                root_certificates = None,
                private_key = None,
                certificate_chain = None):
        print('Init connection from Triton-inference-server')
        print('- Host: {}'.format(triton_host))
        print('- Model: {}'.format(triton_model_name))
        self.triton_host = triton_host
        self.triton_model_name = triton_model_name
        self.model = grpcclient.InferenceServerClient(url = self.triton_host,
                                                    verbose = verbose,
                                                    ssl = ssl,
                                                    root_certificates = root_certificates,
                                                    private_key = private_key,
                                                    certificate_chain = certificate_chain)
        if not self.model.is_server_live():
            print("FAILED : Server not found: {}".format(self.triton_host))
            sys.exit(1)

        if self.triton_model_name is not None:
            if not self.model.is_model_ready(self.triton_model_name):
                print("FAILED : Model not ready: {}".format(self.triton_model_name))
                sys.exit(1)

        self.verbose = verbose

    def run(self,
            data,
            triton_model_name = None,
            meta_inputs = [('input', 'FP32')],
            meta_outputs = [('y', 'FP32'), ('feature', 'FP32')]):
        if self.triton_model_name is None:
            assert triton_model_name is not None, "Current TritonModelGRPC not config model name, please init connection with triton_model_name or specific when call 'run' "
            current_model = triton_model_name
        else:
            current_model = self.triton_model_name
        assert len(data) == len(meta_inputs), "Expect to get {} inputs, your: {}".format(len(inputs), len(data))
        inputs = []
        outputs = []
        if self.verbose:
            tik = time.time()
        for ix, input_tuple in enumerate(meta_inputs):
            inputs.append(grpcclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
            inputs[ix].set_data_from_numpy(data[ix])
            if self.verbose:
                print('Set {} with array {}'.format(meta_inputs[ix], data[ix].shape))
        for ix, output_tuple in enumerate(meta_outputs):
            outputs.append(grpcclient.InferRequestedOutput(output_tuple[0]))

        results = self.model.infer(
            model_name=current_model,
            inputs=inputs,
            outputs=outputs,
            client_timeout=None)
        if self.verbose:
            tok = time.time()
            print('- Time cost:', tok - tik)
        # dict_out = dict()
        net_out = []
        for output_tuple in meta_outputs:
            net_out.append(results.as_numpy(output_tuple[0]))
        return net_out

class RefineTextDet:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = '10.9.3.241:2814', # default gRPC port
                triton_model_name = 'refine_text_det',
                verbose = False,
                ssl = False,
                root_certificates = None,
                private_key = None,
                certificate_chain = None):
        print('Init connection from Triton-inference-server')
        print('- Host: {}'.format(triton_host))
        print('- Model: {}'.format(triton_model_name))
        self.triton_host = triton_host
        self.triton_model_name = triton_model_name
        self.model = grpcclient.InferenceServerClient(url = self.triton_host,
                                                    verbose = verbose,
                                                    ssl = ssl,
                                                    root_certificates = root_certificates,
                                                    private_key = private_key,
                                                    certificate_chain = certificate_chain)
        if not self.model.is_server_live():
            print("FAILED : Server not found: {}".format(self.triton_host))
            sys.exit(1)

        if self.triton_model_name is not None:
            if not self.model.is_model_ready(self.triton_model_name):
                print("FAILED : Model not ready: {}".format(self.triton_model_name))
                sys.exit(1)

        self.verbose = verbose

    def run(self,
            data,
            triton_model_name = None,
            meta_inputs = [('y', 'FP32'), ('feature', 'FP32')],
            meta_outputs = [('y_refiner', 'FP32')]):
        if self.triton_model_name is None:
            assert triton_model_name is not None, "Current TritonModelGRPC not config model name, please init connection with triton_model_name or specific when call 'run' "
            current_model = triton_model_name
        else:
            current_model = self.triton_model_name
        assert len(data) == len(meta_inputs), "Expect to get {} inputs, your: {}".format(len(inputs), len(data))
        inputs = []
        outputs = []
        if self.verbose:
            tik = time.time()
        for ix, input_tuple in enumerate(meta_inputs):
            inputs.append(grpcclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
            inputs[ix].set_data_from_numpy(data[ix])
            if self.verbose:
                print('Set {} with array {}'.format(meta_inputs[ix], data[ix].shape))
        for ix, output_tuple in enumerate(meta_outputs):
            outputs.append(grpcclient.InferRequestedOutput(output_tuple[0]))

        results = self.model.infer(
            model_name=current_model,
            inputs=inputs,
            outputs=outputs,
            client_timeout=None)
        if self.verbose:
            tok = time.time()
            print('- Time cost:', tok - tik)
        # dict_out = dict()
        net_out = []
        for output_tuple in meta_outputs:
            net_out.append(results.as_numpy(output_tuple[0]))
        return net_out