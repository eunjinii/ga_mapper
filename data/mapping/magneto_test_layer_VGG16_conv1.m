Network VGG16 {
  Layer VGG16_layer1 {
    Type: CONV
    Stride { X: 1, Y: 1 }
    Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			SpatialMap(18,18) K;
			TemporalMap(1, 1) Y;
			TemporalMap(3, 3) C;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			Cluster(76, P);
			SpatialMap(110,110) Y;
			TemporalMap(2, 2) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(5, 5) K;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(35, 35) X;
		}
  }
}