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

type sentencePart struct {
	Text       string
	StartIndex int32
}

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

func sendTextToModel(client pb.PravopislyCommsClient, text string) (*pb.ModelReply, error) {
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

func getSuggestionsForText(text string) ([]*pb.TextSuggestion, error) {
	conn, err := grpc.NewClient(
		"localhost:50051",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := pb.NewPravopislyCommsClient(conn)

	allSuggestions := []*pb.TextSuggestion{}
	sentences := splitIntoSentences(text)

	for _, sentence := range sentences {
		reply, err := sendTextToModel(client, sentence.Text)
		if err != nil {
			return nil, err
		}

		for _, suggestion := range reply.GetSuggestions() {
			shiftedSuggestion := &pb.TextSuggestion{
				Type:       suggestion.GetType(),
				StartIndex: suggestion.GetStartIndex() + sentence.StartIndex,
				EndIndex:   suggestion.GetEndIndex() + sentence.StartIndex,
				Replacements: append(
					[]string{},
					suggestion.GetReplacements()...,
				),
			}

			allSuggestions = append(allSuggestions, shiftedSuggestion)
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
		http.Error(w, "failed to get suggestions", http.StatusInternalServerError)
		fmt.Println("model error:", err)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	err = json.NewEncoder(w).Encode(map[string][]*pb.TextSuggestion{
		"suggestions": suggestions,
	})
	if err != nil {
		fmt.Println("json encode error:", err)
	}
}

func main() {
	http.HandleFunc("/api/suggestions", suggestionsHandler)

	fmt.Println("server running on http://localhost:3000")

	err := http.ListenAndServe(":3000", nil)
	if err != nil {
		fmt.Println("err:", err)
		return
	}
}
