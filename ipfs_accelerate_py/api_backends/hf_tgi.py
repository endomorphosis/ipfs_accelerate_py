class hf_tgi:
    def __init__(self, resources, metadata):
        return None
    
    
    
    # def request_tgi_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
    #     incoming_batch_size = len(batch)
    #     endpoint_batch_size = 0
    #     if endpoint in self.endpoint_status:
    #         endpoint_batch_size = self.endpoint_status[endpoint]
    #     elif endpoint_type == None:
    #         for endpoint_type in self.endpoint_types:
    #             if endpoint_type in self.__dict__.keys():
    #                 if model in self.__dict__[endpoint_type]:
    #                     for endpoint in self.__dict__[endpoint_type][model]:
    #                         endpoint_batch_size = self.endpoint_status[endpoint]
    #                         if self.endpoint_status[endpoint] >= incoming_batch_size:
    #                             return endpoint
    #                 else:
    #                     if incoming_batch_size > endpoint_batch_size:
    #                         return ValueError("Batch size too large")
    #                     else:
    #                         return None
    #             else:
    #                 pass
    #     else:
    #         if model in self.__dict__[endpoint_type]:
    #             for endpoint in self.__dict__[endpoint_type][model]:
    #                 endpoint_batch_size = self.endpoint_status[endpoint]
    #                 if self.endpoint_status[endpoint] >= incoming_batch_size:
    #                     return endpoint
    #                 else:
    #                     if incoming_batch_size > endpoint_batch_size:
    #                         return ValueError("Batch size too large")
    #                     else:
    #                         return None
    #         else:
    #             return None
                
    #     if incoming_batch_size > endpoint_batch_size:
    #         return ValueError("Batch size too large")
    #     else:
    #         if model in self.endpoints:
    #             for endpoint in self.tei_endpoints[model]:
    #                 if self.endpoint_status[endpoint] >= incoming_batch_size:
    #                     return endpoint
    #         return None
    #     resources