Network YOLOv8 {
    Layer InceptionA_branch1x1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 192, R: 1, S: 1, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch5x5_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 48, C: 192, R: 1, S: 1, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch5x5_2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 48, R: 5, S: 5, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch3x3dbl_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 192, R: 1, S: 1, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch3x3dbl_2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 96, C: 64, R: 3, S: 3, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch3x3dbl_3 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 96, C: 96, R: 3, S: 3, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer InceptionA_branch_pool {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 192, R: 1, S: 1, Y: 35, X: 35 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer fc {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 1000, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
}
//   (Mixed_5b): InceptionA(
//     (branch1x1): ConvNormAct(
//       (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
//     (branch5x5_1): ConvNormAct(
//       (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
//     (branch5x5_2): ConvNormAct(
//       (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
//     (branch3x3dbl_1): ConvNormAct(
//       (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
//     (branch3x3dbl_2): ConvNormAct(
//       (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
//     (branch3x3dbl_3): ConvNormAct(
//       (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
//     (branch_pool): ConvNormAct(
//       (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
//   )
//   (fc): Linear(in_features=2048, out_features=1000, bias=True)