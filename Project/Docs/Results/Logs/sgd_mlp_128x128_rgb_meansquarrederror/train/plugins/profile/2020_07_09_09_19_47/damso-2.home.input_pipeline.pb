$	�S㥛��@��V�ѫ@�|?5^fd@!D�l�[:�@$	y��'X@P���@�z���V@!39��X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$J+�݌@u�V�E@A)\����B@YP��nY�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&��~j��@bX9���?A��Q�� @Y)\���Ǌ@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�|?5^fd@
ףp=
�?Aףp=
�@Y�����c@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D�l�[:�@�~j�t��?AbX9�HV@Y�n� �@*	   @�*rA2P
Iterator::Model::Prefetch�� ���@!���Xy�H@)�� ���@1���Xy�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�O����@!�F�+wI@)��C��@1G���IF@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator+�h�@!A���.@)+�h�@1A���.@:Preprocessing2F
Iterator::Model�|?5���@!`�Ԉ�H@)
ףp=
�?1���3�>?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor����Mbp?!�+,i�>)����Mbp?1�+,i�>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 98.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	R���&@.]s�}5@�~j�t��?!u�V�E@	!       "	!       *	!       2$	��ʡEBA@y*Q%�C@ףp=
�@!bX9�HV@:	!       B	!       J$	>
ף��@����@�����c@!�n� �@R	!       Z$	>
ף��@����@�����c@!�n� �@JCPU_ONLY