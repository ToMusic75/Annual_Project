$	�/�T��@�ے�=��@X9��v~h@!^�I�#�@$	�+X�xW@o��n8@�<�,��T@!�dZ�X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?5^�I�@�p=
�#_@Aףp=
�I@Y-��麟�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D�l���@-����?Au�V�+@Y�O��np�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9��v~h@V-����?A��K7I @Y�t�Bg@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^�I�#�@D�l����?A���Q�j@Y-�����@*	    �5yA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�~j���@!�'g�I@)�~j���@1�'g�I@:Preprocessing2P
Iterator::Model::Prefetch�Q����@!���&x�H@)�Q����@1���&x�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap7�A`e��@!��N�Z	I@)��"��~@1�l�l�?:Preprocessing2F
Iterator::ModelˡE���@!<�E��H@)����K�?1S�i�V?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�~j�t�x?!=]�����>)�~j�t�x?1=]�����>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 96.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	��v���?@�m7�f�N@D�l����?!�p=
�#_@	!       "	!       *	!       2$	����	R@�/w�bX@��K7I @!���Q�j@:	!       B	!       J$	!�rhQǩ@����i�@�t�Bg@!-�����@R	!       Z$	!�rhQǩ@����i�@�t�Bg@!-�����@JCPU_ONLY