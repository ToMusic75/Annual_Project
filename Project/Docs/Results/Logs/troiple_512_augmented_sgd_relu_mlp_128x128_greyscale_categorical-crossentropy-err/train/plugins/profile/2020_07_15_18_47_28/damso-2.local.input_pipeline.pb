$	���'z�@9&�j�f�@�x�&1�f@!�~j���@$	nb�Х,X@|��Q=�@���4�V@!�*L��X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Zd;_��@�O��nbL@Aq=
ףG@Yd;�O���@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&fffff��@R���Q�?A���S�@Y/�$U�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�x�&1�f@��ʡE�?A��Q��@Y�p=
�kf@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�~j���@333333�?A�A`��rI@Y��Qh��@*	   P �tA2P
Iterator::Model::Prefetch
ףp]'�@!C�����H@)
ףp]'�@1C�����H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapV-5�@!,I8~I@)V-����@1Z���D@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator��MbP�@!��0]��#@)��MbP�@1��0]��#@:Preprocessing2F
Iterator::Model\���h'�@!Զǁ��H@)
ףp=
�?1����1;?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�~j�t�h?!&w���>)�~j�t�h?1&w���>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 98.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	fffffF-@����<@333333�?!�O��nbL@	!       "	!       *	!       2$	�K7�A�:@��M�pL9@��Q��@!�A`��rI@:	!       B	!       J$	�A`�'�@pu��T�@�p=
�kf@!��Qh��@R	!       Z$	�A`�'�@pu��T�@�p=
�kf@!��Qh��@JCPU_ONLY