package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	pb "github.com/YoDobchev/pravopisly/be/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func sendTextToModel(text string) (*pb.ModelReply, error) {
	conn, err := grpc.NewClient(
		"localhost:50051",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := pb.NewPravopislyCommsClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	reply, err := client.SendText(ctx, &pb.SendToModel{
		Text: text,
	})
	if err != nil {
		return nil, err
	}

	return reply, nil
}

type suggestionReq struct {
	Text string
}

func suggestionsHandler(w http.ResponseWriter, r *http.Request) {
	var res suggestionReq

	err := json.NewDecoder(r.Body).Decode(&res)
	if err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
	}
	fmt.Println(res.Text)

	reply, err := sendTextToModel(res.Text)

	json.NewEncoder(w).Encode(map[string][]float32{
		"commaProbs":   reply.GetCommaProbs(),
		"grammarProbs": reply.GetGrammarProbs(),
	})
}

func main() {
	http.HandleFunc("POST /api/suggestions", suggestionsHandler)

	fmt.Println("server running on http://localhost:3000")
	err := http.ListenAndServe(":3000", nil)
	if err != nil {
		fmt.Println("err:", err)
		return
	}
}
