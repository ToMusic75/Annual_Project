$	��n��@������@!�rh��e@!��K7y�@$	 3�eisX@ |�%�@�įUW@!��
��X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-�����@����xIB@A�C�l�[;@Y����x��@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^�Ig�@F����x�?A=
ףp=@YH�z�C�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!�rh��e@��x�&1�?A�� �rh�?Y��(\�Ne@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&��K7y�@���S㥻?Aףp=
�<@YZd;��@*	   �tsA2P
Iterator::Model::Prefetch�C�l'��@![�u�H@)�C�l'��@1[�u�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap���(���@!��8vI@)q=
׳��@1�&)<�H@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorD�l���f@!�V��dN�?)D�l���f@1�V��dN�?:Preprocessing2F
Iterator::Model-���7��@!a�ǉ�H@)P��n��?1����D?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�~j�t�h?!�	�N���>)�~j�t�h?1�	�N���>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 99.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	ʡE��-#@3F0�1@���S㥻?!����xIB@	!       "	!       *	!       2$	�j�t�x.@6w�&|-@�� �rh�?!ףp=
�<@:	!       B	!       J$	��C�L�@%���@��(\�Ne@!Zd;��@R	!       Z$	��C�L�@%���@��(\�Ne@!Zd;��@JCPU_ONLY