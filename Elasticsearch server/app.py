import re
import ast
import os

from flask import Flask, render_template, request, session
from search import Search
from openai import OpenAI

load_dotenv()
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")


app = Flask(__name__)
app.secret_key = 'password' #session 사용할 때 필요한 secret key
es = Search()
client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
)

def extract_filters(query):
    filters = []

    filter_regex = r'category:([^\s]+)\s*' #ex) work from home category:sharepoint
    m = re.search(filter_regex, query)
    if m:
        filters.append({
            'term': {
                'category.keyword': {
                    'value': m.group(1)
                }
            }
        })
        query = re.sub(filter_regex, '', query).strip()

    filter_regex = r'year:([^\s]+)\s*' #ex) work from home year:2020
    m = re.search(filter_regex, query)
    if m:
        filters.append({
            'range': {
                'date': {
                    'gte': f'{m.group(1)}||/y',
                    'lte': f'{m.group(1)}||/y',
                }
            },
        })
        query = re.sub(filter_regex, '', query).strip()

    return {'filter': filters}, query


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/')
def handle_search():
    query = request.form.get('query', '')
    filters, parsed_query = extract_filters(query)
    from_ = request.form.get('from_', type = int, default = 0)
    # return render_template(
    #     'index.html', query=query, results=[], from_=0, total=0) #from_: zero-based index of the first result, total: the total number of results
    
    #multi-match는 search text가 비어있을 때 아무 결과도 반환하지 않음.
    #이 문제를 해결하기 위해 text가 비어있을 때에는 match_all을 사용

    if parsed_query:
        search_query = {
            'must': {
                'multi_match': {
                    'query': parsed_query,
                    'fields': ['question_summary', 'answer_summary'],
                    # 'type': 'phrase_prefix'
                }
            }
        }

    else:
        search_query = {
            'must': {
                'match_all': {}
            }
        }

    results = es.search(
        query={
            'bool': {
                **search_query,
                **filters
            }
        },
        knn={
            'field': 'embedding',
            'query_vector': es.get_embedding(parsed_query),
            'k': 10,
            'num_candidates': 50, #the number of candidate documents to consider from each shard. Elasticsearch retrieves this many candidates from each shard, combines them into a single list and then finds the closest "k" to return as results
            **filters,
        },
        # rank={
        #     'rrf': {}
        # },
        aggs={
            'category-agg': {
                'terms': {
                    'field': 'category.keyword',
                }
            },
            'year-agg': {
                'date_histogram': {
                    'field': 'date',
                    'calendar_interval': 'year',
                    'format': 'yyyy',
                },
            },
        },
        size=10,
        from_=from_
    )
    aggs = {
        'Category': {
            bucket['key']: bucket['doc_count']
            for bucket in results['aggregations']['category-agg']['buckets']
        },
        'Year': {
            bucket['key_as_string']: bucket['doc_count']
            for bucket in results['aggregations']['year-agg']['buckets']
            if bucket['doc_count'] > 0
        },
    }

    #gpt-4o-mini로 rag 답변 생성
    if from_ == 0:
        k = 1
        context = ""
        for hit in results["hits"]["hits"][:k]:
            try: #answer_withurl이 커뮤니티 글인 경우 []로 둘러싸여 있지 않기에 ast.literal_eval 사용 불가, 예외처리
                answers = ast.literal_eval(hit['_source']['answer_withurl'])
            except (ValueError, SyntaxError):
                answers = ast.literal_eval('["' + hit['_source']['answer_withurl'] + '"]')
        
            context = "\n".join(answers)  # 상위 k개 문서 텍스트
        
        #첫 페이지에서 생성된 rag 답변 유지되어야 함
    
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {query}, 짧게 대답해. \n Context: {context}"}
                ],
            max_completion_tokens = 500,
            temperature = 0.5
        )
        session['rag_result'] = completion.choices[0].message.content
    else:
        completion = None



    return render_template('index.html', results=results['hits']['hits'],
                           query=query, from_=from_,
                           total=results['hits']['total']['value'],
                           aggs=aggs,
                           rag_result = session.get('rag_result'),
                           previous_from=max(0, from_ - 10))

@app.get('/document/<id>')
def get_document(id):
    document = es.retrieve_document(id)
    title = document['_source']['question_withurl']
    questioner = document['_source']['questioner']
    # paragraphs = document['_source']['answer'].split('\n')
    # return render_template('document.html', title=title, paragraphs=paragraphs)
    
    
    
    try: #answer_withurl이 커뮤니티 글인 경우 []로 둘러싸여 있지 않기에 ast.literal_eval 사용 불가, 예외처리
        answers = ast.literal_eval(document['_source']['answer_withurl'])
    except (ValueError, SyntaxError):
        answers = ast.literal_eval('["' + document['_source']['answer_withurl'] + '"]')

    try:
        respondents = ast.literal_eval(document['_source']['respondent'])
    except (ValueError, SyntaxError):
        respondents = ast.literal_eval('["' + document['_source']['respondent'] + '"]')
    
    # paragraphs = [f'{respondent} : {answer}' for respondent, answer in zip(respondents, answers)] #닉네임에 실명 나타난 경우가 있어서 주석처리
    
    paragraphs = [f'답변 : {answer}' for respondent, answer in zip(respondents, answers)] #닉네임에 실명 나타난 경우가 있어서 주석처리
    url = document['_source']['url'] if len(document['_source']['url']) > 5 else None
    return render_template('document.html', title=title, questioner = questioner, paragraphs=paragraphs, url = url)

@app.cli.command()
def reindex():
    """Regenerate the Elasticsearch index."""
    response = es.reindex()
    print(f'Index with {len(response["items"])} documents created '
          f'in {response["took"]} milliseconds.')
    

