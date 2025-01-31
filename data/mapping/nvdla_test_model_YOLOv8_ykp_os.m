Network YOLOv8 {
    Layer conv_0 {
        Type: CONV
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer c2f_4_cv1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 160, X: 160 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer c2f_12_cv2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 80, X: 80 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer c2f_18_cv1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer c2f_21_cv2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer detect_22_cv2_0 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 10, X: 10 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
} 

// YOLO(
//   (model): DetectionModel(
//     (model): Sequential(
//       (0): Conv(
//         (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
//       )
//       (4): C2f(
//         (cv1): Conv(
//           (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         )
//       )
//       (12): C2f(
//         (cv2): Conv(
//           (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         )
//       )
//       (18): C2f(
//         (cv1): Conv(
//           (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         )
//       )
//       (21): C2f(
//         (cv2): Conv(
//           (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         )
//       )
//       (22): Detect(
//         (cv2): ModuleList(
//           (0): Sequential(
//             (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
//           )
//         )
//       )
//     )
//   )
// )