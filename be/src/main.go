package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode"

	pb "github.com/YoDobchev/pravopisly/be/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type suggestionReq struct {
	Text string `json:"Text"`
}

type suggestionResp struct {
	Type         int32    `json:"type"`
	StartIndex   int32    `json:"start_index"`
	EndIndex     int32    `json:"end_index"`
	Replacements []string `json:"replacements"`
}

type suggestionsResp struct {
	Suggestions []suggestionResp `json:"suggestions"`
}

type sentencePart struct {
	Text       string
	StartIndex int32
}

var grpcClient pb.PravopislyCommsClient

func isSentenceEnd(r rune) bool {
	return r == '.' || r == '!' || r == '?' || r == '…'
}

func skipSpaces(runes []rune, index int) int {
	for index < len(runes) && unicode.IsSpace(runes[index]) {
		index++
	}

	return index
}

func splitIntoSentences(text string) []sentencePart {
	runes := []rune(text)
	parts := []sentencePart{}

	start := skipSpaces(runes, 0)

	for i := start; i < len(runes); i++ {
		if !isSentenceEnd(runes[i]) {
			continue
		}

		end := i + 1

		for end < len(runes) && isSentenceEnd(runes[end]) {
			end++
		}

		sentenceText := strings.TrimSpace(string(runes[start:end]))

		if sentenceText != "" {
			parts = append(parts, sentencePart{
				Text:       sentenceText,
				StartIndex: int32(start),
			})
		}

		start = skipSpaces(runes, end)
		i = start - 1
	}

	if start < len(runes) {
		sentenceText := strings.TrimSpace(string(runes[start:]))

		if sentenceText != "" {
			parts = append(parts, sentencePart{
				Text:       sentenceText,
				StartIndex: int32(start),
			})
		}
	}

	return parts
}

func sendTextToModel(text string) (*pb.ModelReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	return grpcClient.SendText(ctx, &pb.SendToModel{
		Text: text,
	})
}

func getSuggestionsForText(text string) ([]suggestionResp, error) {
	allSuggestions := []suggestionResp{}
	sentences := splitIntoSentences(text)

	for _, sentence := range sentences {
		reply, err := sendTextToModel(sentence.Text)
		if err != nil {
			return nil, err
		}

		for _, suggestion := range reply.GetSuggestions() {
			allSuggestions = append(allSuggestions, suggestionResp{
				Type:       int32(suggestion.GetType()),
				StartIndex: suggestion.GetStartIndex() + sentence.StartIndex,
				EndIndex:   suggestion.GetEndIndex() + sentence.StartIndex,
				Replacements: append(
					[]string{},
					suggestion.GetReplacements()...,
				),
			})
		}
	}

	return allSuggestions, nil
}

func suggestionsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req suggestionReq

	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	fmt.Println(req.Text)

	suggestions, err := getSuggestionsForText(req.Text)
	if err != nil {
		fmt.Println("model error:", err)
		http.Error(w, "failed to get suggestions", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	err = json.NewEncoder(w).Encode(suggestionsResp{
		Suggestions: suggestions,
	})
	if err != nil {
		fmt.Println("json encode error:", err)
	}
}

func main() {
	conn, err := grpc.NewClient(
		"localhost:50051",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		fmt.Println("grpc connection error:", err)
		return
	}
	defer conn.Close()

	grpcClient = pb.NewPravopislyCommsClient(conn)

	http.HandleFunc("/api/suggestions", suggestionsHandler)

	fmt.Println("server running on http://localhost:3000")

	err = http.ListenAndServe(":3000", nil)
	if err != nil {
		fmt.Println("err:", err)
	}
}
