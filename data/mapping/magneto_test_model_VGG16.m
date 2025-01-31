Network VGG16 {
    Layer Conv_0 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
    Layer Conv_5 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
    Layer Conv_7 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
    Layer Conv_10 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
    Layer Conv_fc1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4096, C: 512, R: 7, S: 7, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
    Layer Linear {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 1000, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(769,769) K;
			TemporalMap(2157, 2157) C;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) Y;
			Cluster(14, P);
			SpatialMap(816,816) C;
			TemporalMap(54, 54) K;
			TemporalMap(1, 1) X;
			TemporalMap(1, 1) Y;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
}