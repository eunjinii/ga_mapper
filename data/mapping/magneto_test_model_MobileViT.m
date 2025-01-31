Network MobileViT {
    Layer sa_query {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
    Layer sa_key {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
    Layer sa_value {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
    Layer sa_output {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
    Layer interm_dense {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
    Layer output_dense {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 1, S: 1, Y: 784, X: 1 }
		Dataflow {
			SpatialMap(38,38) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(64, 64) C;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(8, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(33, 33) C;
			TemporalMap(6, 6) K;
			TemporalMap(1, 1) X;
			TemporalMap(259, 259) Y;
		}
    }
}

// (transformer): MobileViTTransformer(
// (layer): ModuleList(
//     (0-1): 2 x MobileViTTransformerLayer(
//     (attention): MobileViTAttention(
//         (attention): MobileViTSelfAttention(
//         (query): Linear(in_features=64, out_features=64, bias=True)
//         (key): Linear(in_features=64, out_features=64, bias=True)
//         (value): Linear(in_features=64, out_features=64, bias=True)
//         )
//         (output): MobileViTSelfOutput(
//         (dense): Linear(in_features=64, out_features=64, bias=True)
//         )
//     )
//     (intermediate): MobileViTIntermediate(
//         (dense): Linear(in_features=64, out_features=128, bias=True)
//     )
//     (output): MobileViTOutput(
//         (dense): Linear(in_features=128, out_features=64, bias=True)
//     )
// )